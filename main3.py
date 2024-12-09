# File: multi_camera_tracker.py

import cv2
import torch
import numpy as np
from torchvision import transforms
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
from PIL import Image
import asyncio
from db_handler import ReIDDatabase
import torchreid
import torch.nn.functional as F
from collections import defaultdict, deque
import os
import logging
import faiss
import time
import random

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class FeatureExtractor:
    """Extract deep ReID features using a pre-trained model from torchreid."""
    def __init__(self, model_name='osnet_x1_0', feature_dim=512):
        self.feature_dim = feature_dim
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = torchreid.models.build_model(
            name=model_name, num_classes=1000, pretrained=True
        )
        self.model.eval()
        if torch.cuda.is_available():
            self.model = self.model.cuda()

    def extract(self, img_tensor):
        if torch.cuda.is_available():
            img_tensor = img_tensor.to(self.device)
        with torch.no_grad():
            features = self.model(img_tensor)
            features = F.normalize(features, p=2, dim=1)
        return features.cpu().numpy()[0]

class MultiCameraTracker:
    def __init__(self, camera_urls, db_connection_string):
        self.yolo_model = YOLO('yolov8m.pt')
        self.detection_confidence = 0.65
        self.feature_extractor = FeatureExtractor()
        self.db = ReIDDatabase(db_connection_string)

        self.persons = {}  # Map of person ID to historical features
        self.camera_persons = defaultdict(dict)  # Track per-camera person states
        self.recent_global_ids = deque(maxlen=100)  # Avoid duplicate cross-camera IDs
        self.camera_trackers = {i: DeepSort(max_age=30, n_init=3, nms_max_overlap=0.6, max_cosine_distance=0.4)
                                 for i in range(len(camera_urls))}  # Fine-tuned parameters
        self.match_threshold = 0.7
        self.feature_history = defaultdict(lambda: deque(maxlen=10))
        self.camera_urls = camera_urls

        self.door_zones = {}
        self.entry_exit_counts = defaultdict(lambda: {"entry": 0, "exit": 0})

        # FAISS Index for ANN searches
        self.faiss_index = faiss.IndexFlatL2(self.feature_extractor.feature_dim)
        self.person_id_map = {}  # Map FAISS index positions to person IDs

    async def initialize_state_from_db(self):
        """Initialize counts and person features from the database."""
        counts = await self.retry_db_operation(self.db.get_all_counts)
        for camera_id, count_data in counts.items():
            self.entry_exit_counts[camera_id]["entry"] = count_data.get("entry", 0)
            self.entry_exit_counts[camera_id]["exit"] = count_data.get("exit", 0)

        all_persons = await self.retry_db_operation(self.db.get_all_persons)
        for person_id, features in all_persons.items():
            self.persons[int(person_id)] = {
                "features": np.array(features),
                "history": set()
            }
            # Add to FAISS index
            self.faiss_index.add(np.array([features], dtype=np.float32))
            self.person_id_map[self.faiss_index.ntotal - 1] = int(person_id)

    async def retry_db_operation(self, db_function, max_retries=3, delay=2):
        """Retry a database operation with exponential backoff."""
        for attempt in range(max_retries):
            try:
                return db_function()
            except Exception as e:
                if attempt < max_retries - 1:
                    logging.warning(f"Database operation failed. Retrying in {delay} seconds... ({e})")
                    time.sleep(delay * (2 ** attempt))
                else:
                    logging.error("Database operation failed after retries.")
                    raise

    async def update_counts_in_db(self, camera_id):
        """Update counts in the database for a specific camera."""
        await self.retry_db_operation(
            lambda: self.db.update_counts(
                camera_id,
                self.entry_exit_counts[camera_id]["entry"],
                self.entry_exit_counts[camera_id]["exit"]
            )
        )

    def _compute_similarity(self, features):
        """Find the closest person in the FAISS index using cosine similarity."""
        features = features / np.linalg.norm(features)
        distances, indices = self.faiss_index.search(np.array([features], dtype=np.float32), 1)
        return indices[0][0], distances[0][0]

    async def _find_or_create_person(self, features, camera_id):
        """Find a matching person in the FAISS index or create a new one."""
        if features is None:
            return None

        # Normalize features for FAISS search
        features = features / np.linalg.norm(features)
        distances, indices = self.faiss_index.search(np.array([features], dtype=np.float32), 1)

        # Validate FAISS result
        index = indices[0][0]  # Get the first (best) result
        distance = distances[0][0]  # Get the corresponding distance
        if index != -1 and distance <= self.match_threshold:  # Match found
            matched_id = self.person_id_map[index]
            self.persons[matched_id]["history"].add(camera_id)
            if matched_id not in self.recent_global_ids:
                self.recent_global_ids.append(matched_id)
            return matched_id

        # No match found, create a new person ID
        new_id = max(self.persons.keys(), default=0) + 1
        self.feature_history[new_id].append(features)
        self.persons[new_id] = {"features": features, "history": {camera_id}}
        self.faiss_index.add(np.array([features], dtype=np.float32))
        self.person_id_map[self.faiss_index.ntotal - 1] = new_id
        await self.retry_db_operation(lambda: self.db.store_person(new_id, features, camera_id))
        self.recent_global_ids.append(new_id)
        return new_id


    def _preprocess_image(self, person_img):
        """Preprocess image for feature extraction."""
        person_img = cv2.cvtColor(person_img, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(person_img)
        transform = transforms.Compose([
            transforms.Resize((256, 128)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        return transform(pil_img).unsqueeze(0)

    async def process_frame(self, frame, camera_id):
        """Process a single frame from a camera."""
        if camera_id not in self.door_zones:
            frame_height, frame_width = frame.shape[:2]
            self.door_zones[camera_id] = [
                (int(0.3 * frame_width), int(0.4 * frame_height)),
                (int(0.7 * frame_width), int(0.6 * frame_height))
            ]

        detections = []
        results = self.yolo_model.predict(frame, conf=self.detection_confidence, classes=0)

        # Debugging: Log detection count
        logging.info(f"YOLO Detections: {len(results[0].boxes) if results and len(results[0].boxes) > 0 else 0}")

        if results and len(results[0].boxes) > 0:
            for box in results[0].boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                conf = float(box.conf)
                if conf < self.detection_confidence:
                    continue
                w, h = x2 - x1, y2 - y1
                if w * h < 100:  # Ignore too small boxes
                    continue
                detections.append(([x1, y1, w, h], conf, 'person'))

                # Debugging: Log bounding box details
                logging.info(f"Bounding Box: {x1, y1, x2, y2} | Confidence: {conf}")

        # Process tracks using DeepSort
        tracks = self.camera_trackers[camera_id].update_tracks(detections, frame=frame)
        for track in tracks:
            if not track.is_confirmed():
                continue

            x1, y1, x2, y2 = map(int, track.to_tlbr())
            # Debugging: Log track details
            logging.info(f"Track: ID={track.track_id}, BBox={x1, y1, x2, y2}")

            # Draw bounding box and ID
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"ID: {track.track_id}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # Draw door zone
        (x_min, y_min), (x_max, y_max) = self.door_zones[camera_id]
        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 0, 255), 2)
        cv2.putText(frame, "Door Zone", (x_min, y_min - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        # Draw entry/exit counts
        entry_count = self.entry_exit_counts[camera_id]["entry"]
        exit_count = self.entry_exit_counts[camera_id]["exit"]
        cv2.putText(frame, f"Entry: {entry_count} | Exit: {exit_count}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        return frame


    async def run(self):
        """Run the tracker for all cameras."""
        await self.initialize_state_from_db()
        caps = [cv2.VideoCapture(url) for url in self.camera_urls]
        try:
            while True:
                frames = [cap.read()[1] for cap in caps]
                tasks = [self.process_frame(frame, i) for i, frame in enumerate(frames) if frame is not None]
                processed_frames = await asyncio.gather(*tasks)
                for i, frame in enumerate(processed_frames):
                    if frame is not None:
                        cv2.imshow(f"Camera {i}", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        finally:
            for cap in caps:
                cap.release()
            cv2.destroyAllWindows()
        
if __name__ == "__main__":
    camera_urls = [ r'rtsp://Yectra:Yectra123@192.168.1.15:554/stream2']
    # camera_urls = ['cam-1M.mp4','cam-2M.mp4']
    db_connection_string = "AccountEndpoint=https://occupancytrackerdb.documents.azure.com:443/;AccountKey=NTTvzWNTTmZ3I0rydqqnIIjPDGG5RxXVCYa9WS78XK4PvUXUGCS9Tx9s8xnfs4rSfS2xD2deHAGUACDbIMdVxA==;"
    tracker = MultiCameraTracker(camera_urls, db_connection_string)
    asyncio.run(tracker.run())

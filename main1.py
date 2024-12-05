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
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

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
        self.match_threshold = 0.7  # Increased threshold for stricter matching
        self.feature_history = defaultdict(lambda: deque(maxlen=10))
        self.camera_urls = camera_urls

        self.door_zones = {}  # Dynamically defined based on frame size
        self.entry_exit_counts = defaultdict(lambda: {"entry": 0, "exit": 0})

    async def initialize_state_from_db(self):
        """Initialize counts and person features from the database."""
        counts = self.db.get_all_counts()
        for camera_id, count_data in counts.items():
            self.entry_exit_counts[camera_id]["entry"] = count_data.get("entry", 0)
            self.entry_exit_counts[camera_id]["exit"] = count_data.get("exit", 0)

        all_persons = self.db.get_all_persons()
        for person_id, features in all_persons.items():
            self.persons[int(person_id)] = {
                "features": np.array(features),
                "history": set()
            }

    async def update_counts_in_db(self, camera_id):
        """Update counts in the database for a specific camera."""
        self.db.update_counts(
            camera_id,
            self.entry_exit_counts[camera_id]["entry"],
            self.entry_exit_counts[camera_id]["exit"]
        )

    def is_inside_zone(self, center, door_zone):
        """Check if a point is inside a given door zone with a margin."""
        (x_min, y_min), (x_max, y_max) = door_zone
        x, y = center
        margin = 10  # Add margin to reduce false positives
        return x_min - margin <= x <= x_max + margin and y_min - margin <= y <= y_max + margin

    def _compute_similarity(self, features1, features2):
        """Compute cosine similarity between two normalized feature vectors."""
        if not (np.linalg.norm(features1) and np.linalg.norm(features2)):
            return 0  # Avoid division by zero
        features1 = features1 / np.linalg.norm(features1)
        features2 = features2 / np.linalg.norm(features2)
        return np.dot(features1, features2)

    async def _find_or_create_person(self, features, camera_id):
        """Find a matching person in the database or create a new one."""
        if features is None:
            return None

        best_match_id = None
        best_match_score = -1

        # Search for the best match among known persons
        for person_id, person_data in self.persons.items():
            stored_features = person_data["features"]
            similarity = self._compute_similarity(features, stored_features)
            if similarity > best_match_score:
                best_match_score = similarity
                best_match_id = person_id

        # Check match threshold
        if best_match_score > self.match_threshold:
            self.persons[best_match_id]["history"].add(camera_id)
            if best_match_id not in self.recent_global_ids:
                self.recent_global_ids.append(best_match_id)
            return best_match_id

        # If no match is found, create a new ID
        new_id = max(self.persons.keys(), default=0) + 1
        self.feature_history[new_id].append(features)
        self.persons[new_id] = {"features": features, "history": {camera_id}}
        self.db.store_person(new_id, features, camera_id)
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
                # Dynamically define door zone
                frame_height, frame_width = frame.shape[:2]
                self.door_zones[camera_id] = [
                    (int(0.3 * frame_width), int(0.4 * frame_height)),
                    (int(0.7 * frame_width), int(0.6 * frame_height))
                ]

            detections = []
            results = self.yolo_model.predict(frame, conf=self.detection_confidence, classes=0)
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

            tracks = self.camera_trackers[camera_id].update_tracks(detections, frame=frame)
            for track in tracks:
                if not track.is_confirmed():
                    continue

                # Extract ReID features
                x1, y1, x2, y2 = map(int, track.to_tlbr())
                person_img = frame[y1:y2, x1:x2]
                if person_img.size == 0:
                    continue

                person_img_tensor = self._preprocess_image(person_img)
                features = self.feature_extractor.extract(person_img_tensor)
                if features is None or features.size == 0:
                    continue

                global_id = await self._find_or_create_person(features, camera_id)
                center = ((x1 + x2) // 2, (y1 + y2) // 2)

                # Process entry/exit logic
                door_zone = self.door_zones[camera_id]
                previous_position = self.camera_persons[camera_id].get(global_id, {}).get("last_position")
                self.camera_persons[camera_id][global_id] = {"last_position": center}
                if previous_position:
                    in_zone = self.is_inside_zone(center, door_zone)
                    was_in_zone = self.is_inside_zone(previous_position, door_zone)
                    if not was_in_zone and in_zone:  # Entry event
                        self.entry_exit_counts[camera_id]["entry"] += 1
                        await self.update_counts_in_db(camera_id)
                    elif was_in_zone and not in_zone:  # Exit event
                        self.entry_exit_counts[camera_id]["exit"] += 1
                        await self.update_counts_in_db(camera_id)

                # Draw bounding box and display ID
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"ID: {global_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            # Draw door zone and counts
            (x_min, y_min), (x_max, y_max) = self.door_zones[camera_id]
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 0, 255), 2)
            cv2.putText(frame, "Door Zone", (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

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
    camera_urls = ['video5.mp4']
    # camera_urls = ['cam-1M.mp4','cam-2M.mp4']
    db_connection_string = "AccountEndpoint=https://occupancytrackerdb.documents.azure.com:443/;AccountKey=NTTvzWNTTmZ3I0rydqqnIIjPDGG5RxXVCYa9WS78XK4PvUXUGCS9Tx9s8xnfs4rSfS2xD2deHAGUACDbIMdVxA==;"
    tracker = MultiCameraTracker(camera_urls, db_connection_string)
    asyncio.run(tracker.run())

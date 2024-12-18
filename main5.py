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
import faiss
import os
import json

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

class FeatureExtractor:
    """Extract deep ReID features using a pre-trained model from torchreid."""
    def __init__(self, model_name='osnet_ain_x1_0', feature_dim=512):
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
    def __init__(self, camera_details_json, db_connection_string):
        camera_details = json.loads(camera_details_json)
        self.camera_urls = []
        self.camera_positions = {}  # Map camera_id to position
        self.door_zones = {}  # Map camera_id to door zone coordinates
        self.yolo_model = YOLO('yolov8m.pt')
        self.detection_confidence = 0.7
        self.feature_extractor = FeatureExtractor()
        self.db = ReIDDatabase(db_connection_string)
        self.smoothing_alpha = 0.8 

        self.persons = {}  # Map of person ID to historical features
        self.camera_persons = defaultdict(dict)
        self.recent_global_ids = deque(maxlen=100)
        self.camera_trackers = {}  # Map camera_id to DeepSort instances
        self.match_threshold = 0.75  # Stricter threshold
        self.feature_history = defaultdict(lambda: deque(maxlen=10))

        self.faiss_index = faiss.IndexHNSWFlat(self.feature_extractor.feature_dim, 32)
        self.faiss_index.hnsw.efSearch = 50
        self.faiss_index.hnsw.efConstruction = 40
        # self.faiss_index = faiss.index_cpu_to_all_gpus(self.faiss_index)

        self.person_id_map = {}  # Map FAISS index positions to person IDs
        self.entry_exit_counts = defaultdict(lambda: {"entry": 0, "exit": 0})

        self._initialize_cameras(camera_details)

    def _initialize_cameras(self, camera_details):
        """Parse camera details from JSON and initialize trackers and zones."""
        self.index_to_camera_id = {}
        for i, camera in enumerate(camera_details['cameraDetails']):
            entrance_name = camera['entranceName']
            camera_id = entrance_name  # Use entrance name as the camera ID
            video_url = camera['videoUrl']
            door_coords = camera['doorCoordinates']
            camera_position = camera['cameraPosition']

            # Store camera-specific details
            self.camera_urls.append(video_url)
            self.camera_positions[camera_id] = camera_position
            self.door_zones[camera_id] = door_coords
            self.camera_trackers[camera_id] = DeepSort(max_age=50, n_init=5, nms_max_overlap=0.6, max_cosine_distance=0.4)
            self.index_to_camera_id[i] = camera_id
    def _smooth_features(self, old_features, new_features):
                """Apply exponential moving average to smooth feature vectors."""
                if old_features is None:
                    return new_features
                return self.smoothing_alpha * old_features + (1 - self.smoothing_alpha) * new_features

    async def initialize_state_from_db(self):
        """Initialize the FAISS index and local state from the database."""
        counts = self.db.get_all_counts()
        for camera_id, count_data in counts.items():
            self.entry_exit_counts[camera_id]["entry"] = count_data.get("entry", 0)
            self.entry_exit_counts[camera_id]["exit"] = count_data.get("exit", 0)  

        all_persons = self.db.get_all_persons()
        for person_id, features in all_persons.items():
            person_id = int(person_id)
            features = np.array(features, dtype=np.float32)

            # Add features to FAISS
            features = features / np.linalg.norm(features)
            self.faiss_index.add(np.array([features], dtype=np.float32))
            self.person_id_map[self.faiss_index.ntotal - 1] = person_id

            # Update local state
            self.persons[person_id] = {"features": features, "history": set()}

    def _compute_similarity(self, features):
        """Find the most similar feature vector in the FAISS index."""
        features = features / np.linalg.norm(features)  # Normalize input features
        distances, indices = self.faiss_index.search(np.array([features], dtype=np.float32), 5)
        for i, distance in enumerate(distances[0]):
            candidate_index = indices[0][i]
            candidate_person_id = self.person_id_map.get(candidate_index)
            if candidate_person_id is not None:
                db_features = np.array(self.db.get_person_features(candidate_person_id))
                db_features = db_features / np.linalg.norm(db_features)  # Normalize DB features
                cosine_similarity = np.dot(features, db_features.T)
                if cosine_similarity > self.match_threshold:
                    return candidate_index, cosine_similarity
        return -1, float("inf")

    async def _find_or_create_person(self, features, camera_id):
            """Find a matching person using FAISS or create a new one."""
            if features is None:
                return None

            global_id, similarity = self._compute_similarity(features)
            if global_id >= 0:
                person_id = self.person_id_map[global_id]
                self.persons[person_id]["history"].add(camera_id)
                self.persons[person_id]["features"] = self._smooth_features(
                    self.persons[person_id].get("features"), features
                )
                if person_id not in self.recent_global_ids:
                    self.recent_global_ids.append(person_id)
                return person_id

            # Create a new person ID
            new_id = max(self.persons.keys(), default=0) + 1
            smoothed_features = features / np.linalg.norm(features)
            self.feature_history[new_id].append(smoothed_features)
            self.faiss_index.add(np.array([smoothed_features], dtype=np.float32))
            self.person_id_map[self.faiss_index.ntotal - 1] = new_id

            self.db.store_person(new_id, smoothed_features.tolist(), camera_id)
            self.persons[new_id] = {"features": smoothed_features, "history": {camera_id}}
            self.recent_global_ids.append(new_id)
            return new_id

    def is_inside_zone(self, center, door_zone):
        """Check if a point is inside a given door zone with a margin."""
        (x_min, y_min), (x_max, y_max) = door_zone
        x, y = center
        margin = 10
        return x_min - margin <= x <= x_max + margin and y_min - margin <= y <= y_max + margin

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
        detections = []
        results = self.yolo_model.predict(frame, conf=self.detection_confidence, classes=0)
        if results and len(results[0].boxes) > 0:
            for box in results[0].boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                conf = float(box.conf)
                if conf < self.detection_confidence:
                    continue
                detections.append(([x1, y1, x2 - x1, y2 - y1], conf, 'person'))

        tracks = self.camera_trackers[camera_id].update_tracks(detections, frame=frame)
        for track in tracks:
            if not track.is_confirmed():
                continue

            x1, y1, x2, y2 = map(int, track.to_tlbr())
            person_img = frame[y1:y2, x1:x2]
            if person_img.size == 0:
                continue

            person_img_tensor = self._preprocess_image(person_img)
            features = self.feature_extractor.extract(person_img_tensor)
            if features is None or features.size == 0:
                continue

            # Attempt to find or create a person ID
            person_id = await self._find_or_create_person(features, camera_id)
            if person_id is None:
                continue  # Skip processing if person_id could not be created

            # Aggregate features and calculate mean
            self.feature_history[person_id].append(features)
            aggregated_features = np.mean(self.feature_history[person_id], axis=0)

            # Update camera-specific tracking state
            center = ((x1 + x2) // 2, (y1 + y2) // 2)
            door_zone = self.door_zones[camera_id]
            position = self.camera_positions[camera_id]

            previous_position = self.camera_persons[camera_id].get(person_id, {}).get("last_position")
            self.camera_persons[camera_id][person_id] = {"last_position": center}
            if previous_position:
                in_zone = self.is_inside_zone(center, door_zone)
                was_in_zone = self.is_inside_zone(previous_position, door_zone)
                if position == 'inside-out':
                    if not was_in_zone and in_zone:
                        self.entry_exit_counts[camera_id]["exit"] += 1
                        await self.update_counts_in_db(camera_id)
                    elif was_in_zone and not in_zone:
                        self.entry_exit_counts[camera_id]["entry"] += 1
                        await self.update_counts_in_db(camera_id)
                elif position == 'outside-in':
                    if not was_in_zone and in_zone:
                        self.entry_exit_counts[camera_id]["entry"] += 1
                        await self.update_counts_in_db(camera_id)
                    elif was_in_zone and not in_zone:
                        self.entry_exit_counts[camera_id]["exit"] += 1
                        await self.update_counts_in_db(camera_id)

            # Draw bounding box and display ID
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"ID: {person_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # Draw door zone and counts
        (x_min, y_min), (x_max, y_max) = self.door_zones[camera_id]
        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 0, 255), 2)
        cv2.putText(frame, "Door Zone", (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        entry_count = self.entry_exit_counts[camera_id]["entry"]
        exit_count = self.entry_exit_counts[camera_id]["exit"]
        cv2.putText(frame, f"Entry: {entry_count} | Exit: {exit_count}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        return frame


    async def update_counts_in_db(self, camera_id):
        """Update counts in the database for a specific camera."""
        self.db.update_counts(
            camera_id,
            self.entry_exit_counts[camera_id]["entry"],
            self.entry_exit_counts[camera_id]["exit"]
        )

    async def run(self):
        """Run the tracker for all cameras."""
        await self.initialize_state_from_db()
        caps = [cv2.VideoCapture(url) for url in self.camera_urls]
        try:
            while True:
                frames = [cap.read()[1] for cap in caps]
                tasks = [
                    self.process_frame(frame, self.index_to_camera_id[i])
                    for i, frame in enumerate(frames) if frame is not None
                ]
                processed_frames = await asyncio.gather(*tasks)
                for i, frame in enumerate(processed_frames):
                    if frame is not None:
                        cv2.imshow(f"Camera {self.index_to_camera_id[i]}", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        finally:
            for cap in caps:
                cap.release()
            cv2.destroyAllWindows()

async def main():

    db_connection_string = "AccountEndpoint=https://occupancytrackerdb.documents.azure.com:443/;AccountKey=NTTvzWNTTmZ3I0rydqqnIIjPDGG5RxXVCYa9WS78XK4PvUXUGCS9Tx9s8xnfs4rSfS2xD2deHAGUACDbIMdVxA==;"
    db_handler = ReIDDatabase(db_connection_string)
    camera_setup_details = db_handler.get_camera_setup_details()
    if not camera_setup_details:
        print("Failed to fetch camera setup details from the database. Exiting...")
        return
    camera_details_json = json.dumps({
        "cameraDetails": camera_setup_details["cameraDetails"]
    })
    tracker = MultiCameraTracker(camera_details_json, db_connection_string)
    await tracker.run()

if __name__ == "__main__":
    asyncio.run(main())
         
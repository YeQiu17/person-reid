import cv2
import torch
import numpy as np
from torchvision import transforms
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
from PIL import Image
import asyncio
from db_handler1 import ReIDDatabase
import torchreid
import torch.nn.functional as F
from collections import defaultdict, deque
import os
import logging

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
        self.yolo_model = YOLO('yolov8m-pose.pt')  # Use YOLOv8 with pose model
        self.detection_confidence = 0.7
        self.feature_extractor = FeatureExtractor()

        self.db = ReIDDatabase(db_connection_string)
        self.persons = {}  # Map of person ID to historical features
        self.camera_persons = defaultdict(dict)  # Track per-camera person states (inside/outside)
        self.recent_global_ids = deque(maxlen=100)  # Avoid duplicate cross-camera IDs
        self.camera_trackers = {i: DeepSort(max_age=30, n_init=3, nms_max_overlap=1.0, max_cosine_distance=0.3)
                                 for i in range(len(camera_urls))}
        self.match_threshold = 0.75
        self.feature_history = defaultdict(lambda: deque(maxlen=10))
        self.camera_urls = camera_urls

        # Define entry/exit zones for cameras (camera_id -> door rectangle)
        self.door_zones = {
            0: [(342, 226), (906, 568)]  # Example door zone for camera 0
        }

        # Track entry/exit counts for each camera
        self.entry_exit_counts = defaultdict(lambda: {"entry": 0, "exit": 0})

    async def initialize_state_from_db(self):
        """Initialize counts and person features from the database."""
        counts = await self.db.get_all_counts()
        for camera_id, count_data in counts.items():
            self.entry_exit_counts[camera_id]["entry"] = count_data.get("entry", 0)
            self.entry_exit_counts[camera_id]["exit"] = count_data.get("exit", 0)

        all_persons = await self.db.get_all_persons()
        for person_id, features in all_persons.items():
            self.persons[int(person_id)] = {
                "features": np.array(features),
                "history": set()
            }

    async def update_counts_in_db(self, camera_id):
        """Update counts in the database for a specific camera."""
        await self.db.update_counts(
            camera_id,
            self.entry_exit_counts[camera_id]["entry"],
            self.entry_exit_counts[camera_id]["exit"]
        )

    def is_inside_zone(self, point, door_zone):
        """Check if a point (e.g., feet keypoint) is inside a given door zone."""
        (x_min, y_min), (x_max, y_max) = door_zone
        x, y = point
        return x_min <= x <= x_max and y_min <= y <= y_max

    def _compute_similarity(self, features1, features2):
        """Compute cosine similarity between two feature vectors."""
        return np.dot(features1, features2)

    async def _find_or_create_person(self, features, camera_id):
        """Find a matching person in the database or create a new one."""
        if features is None:
            return None

        for person_id, person_data in self.persons.items():
            stored_features = person_data["features"]
            similarity = self._compute_similarity(features, stored_features)
            if similarity > self.match_threshold:
                # Update person's features with average
                self.feature_history[person_id].append(features)
                avg_features = np.mean(self.feature_history[person_id], axis=0)
                self.persons[person_id]["features"] = avg_features
                self.persons[person_id]["history"].add(camera_id)
                await self.db.update_person_features(person_id, avg_features, camera_id)
                return person_id

        # If no match is found, create a new ID
        new_id = max(self.persons.keys(), default=0) + 1
        self.feature_history[new_id].append(features)
        self.persons[new_id] = {"features": features, "history": {camera_id}}
        await self.db.store_person(new_id, features, camera_id)
        return new_id

    async def process_frame(self, frame, camera_id):
        """Process a single frame from a camera, detect entry/exit events."""
        results = self.yolo_model.predict(frame, conf=self.detection_confidence, classes=0)
        if len(results[0].keypoints) > 0:
            for detection, keypoints in zip(results[0].boxes, results[0].keypoints):
                bbox = detection.xyxy[0].cpu().numpy()
                x1, y1, x2, y2 = map(int, bbox)

                # Ensure the keypoints index is valid
                 # Correctly index into keypoints
                if keypoints.shape[0] > 0 and keypoints.shape[1] > 15:  # Ensure valid indices
                    feet_point = keypoints[0, 15].cpu().numpy()  # Correct indexing
                    if feet_point.shape == (2,):  # Validate shape
                        feet_x, feet_y = map(int, feet_point)
                    else:
                        logging.error(f"Unexpected feet_point shape: {feet_point.shape}")
                        continue
                else:
                    logging.error(f"Keypoints do not have the required index: {keypoints.shape}")
                    continue

                person_img = frame[y1:y2, x1:x2]
                if person_img.size == 0:
                    continue

                person_img_tensor = self._preprocess_image(person_img)
                features = self.feature_extractor.extract(person_img_tensor)
                global_id = await self._find_or_create_person(features, camera_id)

                door_zone = self.door_zones[camera_id]
                previous_position = self.camera_persons[camera_id].get(global_id, {}).get("last_position")
                self.camera_persons[camera_id][global_id] = {"last_position": (feet_x, feet_y)}

                if previous_position:
                    in_zone = self.is_inside_zone((feet_x, feet_y), door_zone)
                    was_in_zone = self.is_inside_zone(previous_position, door_zone)

                    if not was_in_zone and in_zone:
                        self.entry_exit_counts[camera_id]["entry"] += 1
                        await self.update_counts_in_db(camera_id)
                    elif was_in_zone and not in_zone:
                        self.entry_exit_counts[camera_id]["exit"] += 1
                        await self.update_counts_in_db(camera_id)

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"ID: {global_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        entry_count = self.entry_exit_counts[camera_id]["entry"]
        exit_count = self.entry_exit_counts[camera_id]["exit"]
        cv2.putText(frame, f"Entry: {entry_count} | Exit: {exit_count}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        return frame

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
    db_connection_string = "AccountEndpoint=https://occupancytrackerdb.documents.azure.com:443/;AccountKey=NTTvzWNTTmZ3I0rydqqnIIjPDGG5RxXVCYa9WS78XK4PvUXUGCS9Tx9s8xnfs4rSfS2xD2deHAGUACDbIMdVxA==;"
    tracker = MultiCameraTracker(camera_urls, db_connection_string)
    asyncio.run(tracker.run())

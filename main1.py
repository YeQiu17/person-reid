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

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


class FeatureExtractor:
    """Extract deep ReID features using a pre-trained model from torchreid."""
    def __init__(self, model_name='osnet_x1_0', feature_dim=512):
        self.feature_dim = feature_dim
        self.model = torchreid.models.build_model(
            name=model_name, num_classes=1000, pretrained=True
        )
        self.model.eval()
        if torch.cuda.is_available():
            self.model = self.model.cuda()

    def extract(self, img_tensor):
        if torch.cuda.is_available():
            img_tensor = img_tensor.cuda()
        with torch.no_grad():
            features = self.model(img_tensor)
            features = F.normalize(features, p=2, dim=1)
        return features.cpu().numpy()[0]


class MultiCameraTracker:
    def __init__(self, camera_urls, db_connection_string):
        self.yolo_model = YOLO('yolov8s.pt')
        self.detection_confidence = 0.6
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

    def is_inside_zone(self, center, door_zone):
        """Check if a point is inside a given door zone."""
        (x_min, y_min), (x_max, y_max) = door_zone
        x, y = center
        return x_min <= x <= x_max and y_min <= y <= y_max

    def _compute_similarity(self, features1, features2):
        """Compute cosine similarity between two feature vectors."""
        return np.dot(features1, features2)

    def _average_features(self, features_list):
        """Compute the average feature vector from a list of features."""
        return np.mean(features_list, axis=0)

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
            # Update historical features in memory
            self.feature_history[best_match_id].append(features)
            avg_features = self._average_features(self.feature_history[best_match_id])
            self.persons[best_match_id]["features"] = avg_features
            self.persons[best_match_id]["history"].add(camera_id)
            if best_match_id not in self.recent_global_ids:
                self.recent_global_ids.append(best_match_id)
            # Update the database with the new average features
            await self.db.update_person_features(best_match_id, avg_features, camera_id)
            return best_match_id

        # If no match is found, create a new ID
        new_id = max(self.persons.keys(), default=0) + 1
        self.feature_history[new_id].append(features)
        self.persons[new_id] = {"features": features, "history": {camera_id}}
        await self.db.store_person(new_id, features, camera_id)
        self.recent_global_ids.append(new_id)
        return new_id

    async def process_frame(self, frame, camera_id):
        """Process a single frame from a camera, detect entry/exit events."""
        detections = []
        results = self.yolo_model.predict(frame, conf=self.detection_confidence, classes=0)
        if len(results[0].boxes) > 0:
            for box in results[0].boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                conf = float(box.conf)
                w, h = x2 - x1, y2 - y1
                detections.append(([x1, y1, w, h], conf, 'person'))

        tracks = self.camera_trackers[camera_id].update_tracks(detections, frame=frame)
        for track in tracks:
            if not track.is_confirmed():
                continue

            x1, y1, x2, y2 = map(int, track.to_tlbr())
            frame_height, frame_width = frame.shape[:2]
            if x1 < 0 or y1 < 0 or x2 > frame_width or y2 > frame_height:
                continue

            person_img = frame[y1:y2, x1:x2]
            if person_img.size == 0:
                continue

            # Extract ReID features
            person_img_tensor = self._preprocess_image(person_img)
            features = self.feature_extractor.extract(person_img_tensor)
            global_id = await self._find_or_create_person(features, camera_id)

            # Get the center of the bounding box
            center = ((x1 + x2) // 2, (y1 + y2) // 2)
            door_zone = self.door_zones[camera_id]

            previous_position = self.camera_persons[camera_id].get(global_id, {}).get("last_position")
            self.camera_persons[camera_id][global_id] = {"last_position": center}

            if previous_position:
                in_zone = self.is_inside_zone(center, door_zone)
                was_in_zone = self.is_inside_zone(previous_position, door_zone)

                # Entering the zone
                if not was_in_zone and in_zone:
                    self.entry_exit_counts[camera_id]["entry"] += 1
                    await self.update_counts_in_db(camera_id)

                # Exiting the zone
                elif was_in_zone and not in_zone:
                    self.entry_exit_counts[camera_id]["exit"] += 1
                    await self.update_counts_in_db(camera_id)

            # Draw bounding box and display ID
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"ID: {global_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # Draw door zone and counts
        if camera_id in self.door_zones:
            (x_min, y_min), (x_max, y_max) = self.door_zones[camera_id]
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 0, 255), 2)
            cv2.putText(frame, "Door Zone", (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

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
    db_connection_string = "AccountEndpoint=https://occupancytrackerdb.documents.azure.com:443/;AccountKey=YOUR_KEY;"
    tracker = MultiCameraTracker(camera_urls, db_connection_string)
    asyncio.run(tracker.run())

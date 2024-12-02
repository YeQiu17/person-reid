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
        self.yolo_model = YOLO('yolov8s-pose.pt')  # Use YOLOv8 Pose model
        self.detection_confidence = 0.6
        self.feature_extractor = FeatureExtractor()
        self.feature_dim = self.feature_extractor.feature_dim

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

    async def load_counts_from_db(self):
        """Load existing counts from the database into memory for each camera."""
        counts = await self.db.get_all_counts()  # Fetch counts for all cameras from the DB
        for camera_id, count_data in counts.items():
            self.entry_exit_counts[camera_id]["entry"] = count_data.get("entry", 0)
            self.entry_exit_counts[camera_id]["exit"] = count_data.get("exit", 0)
        print(f"Counts loaded from DB: {self.entry_exit_counts}")

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

    def _calculate_movement_direction(self, previous_position, current_position):
        """Calculate movement direction based on the change in position."""
        prev_x, prev_y = previous_position
        curr_x, curr_y = current_position
        return (curr_x - prev_x, curr_y - prev_y)

    async def load_features_from_db(self):
        """Load existing features from the database into memory for matching."""
        all_persons = await self.db.get_all_persons()
        self.persons = {person_id: {"features": np.array(features), "history": set()} 
                        for person_id, features in all_persons.items()}

    def _extract_features(self, person_img):
        """Extract deep features for person re-identification."""
        person_img = cv2.cvtColor(person_img, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(person_img)
        transform = transforms.Compose([
            transforms.Resize((256, 128)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        img_tensor = transform(pil_img).unsqueeze(0)
        return self.feature_extractor.extract(img_tensor)

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
            # Match found: directly return the ID without updating features
            return best_match_id

        # If no match is found, create a new ID
        new_id = str(len(self.persons) + 1)
        self.feature_history[new_id].append(features)
        self.persons[new_id] = {"features": features, "history": {camera_id}}
        self.db.store_person(new_id, features, camera_id)  # Only store for new persons
        self.recent_global_ids.append(new_id)
        return new_id

    async def process_frame(self, frame, camera_id):
        """Process a single frame from a camera, detect entry/exit events using poses."""
        results = self.yolo_model.predict(frame, conf=self.detection_confidence, classes=0)

        # Ensure results contain keypoints (pose data)
        if results[0].keypoints is not None:
            keypoints_list = results[0].keypoints.numpy()  # Convert keypoints to a NumPy array
            confidences = results[0].boxes.conf.numpy() if results[0].boxes else []  # Confidence scores

            for keypoints, conf in zip(keypoints_list, confidences):
                # Validate keypoints: Ensure they are a valid 2D array
                if not isinstance(keypoints, np.ndarray) or keypoints.ndim != 2 or keypoints.shape[1] != 2:
                    print(f"Invalid keypoints shape: {keypoints.shape if isinstance(keypoints, np.ndarray) else 'unknown'}")
                    continue

                # Calculate pose center from keypoints
                x_coords = keypoints[:, 0]  # Extract x-coordinates
                y_coords = keypoints[:, 1]  # Extract y-coordinates
                center = (int(np.mean(x_coords)), int(np.mean(y_coords)))

                # Extract cropped person image for feature extraction
                x_min, y_min = int(min(x_coords)), int(min(y_coords))
                x_max, y_max = int(max(x_coords)), int(max(y_coords))
                person_img = frame[y_min:y_max, x_min:x_max]
                if person_img.size == 0:
                    continue

                # Extract features and match ID
                features = self._extract_features(person_img)
                global_id = await self._find_or_create_person(features, camera_id)

                # Get previous position of the person
                previous_position = self.camera_persons[camera_id].get(global_id, {}).get("last_position")
                self.camera_persons[camera_id][global_id] = {"last_position": center}

                # Determine entry/exit events based on position and zone
                door_zone = self.door_zones[camera_id]
                in_zone = self.is_inside_zone(center, door_zone)
                was_in_zone = self.is_inside_zone(previous_position, door_zone) if previous_position else False

                if previous_position:
                    if not was_in_zone and in_zone:  # Entering the zone
                        self.entry_exit_counts[camera_id]["entry"] += 1
                        self.camera_persons[camera_id][global_id]["state"] = "inside"
                        await self.update_counts_in_db(camera_id)

                    elif was_in_zone and not in_zone:  # Exiting the zone
                        self.entry_exit_counts[camera_id]["exit"] += 1
                        self.camera_persons[camera_id][global_id]["state"] = "outside"
                        await self.update_counts_in_db(camera_id)

                # Draw pose keypoints and center point on the frame
                for x, y in zip(x_coords, y_coords):
                    cv2.circle(frame, (int(x), int(y)), 3, (0, 255, 0), -1)  # Green keypoints
                cv2.circle(frame, center, 5, (255, 0, 0), -1)  # Blue center

        # Draw door zone and counts
        if camera_id in self.door_zones:
            (x_min, y_min), (x_max, y_max) = self.door_zones[camera_id]
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 0, 255), 2)  # Red box for the door zone
            cv2.putText(frame, "Door Zone", (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        # Overlay entry/exit counts on the video frame
        entry_count = self.entry_exit_counts[camera_id]["entry"]
        exit_count = self.entry_exit_counts[camera_id]["exit"]
        cv2.putText(frame, f"Entry: {entry_count} | Exit: {exit_count}", 
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        return frame


    async def run(self):
        """Run the tracker for all cameras."""
        await self.load_features_from_db()
        await self.load_counts_from_db()  # Load counts from the database
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

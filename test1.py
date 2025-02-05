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
import logging
import json
from dotenv import load_dotenv
from scipy.optimize import linear_sum_assignment

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
load_dotenv()

class LocalTracker:
    """Handle local tracking using Hungarian algorithm."""
    def __init__(self, max_age=50, min_similarity=0.6):
        self.max_age = max_age
        self.min_similarity = min_similarity
        self.tracks = {}
        self.age = defaultdict(int)
        
    def compute_cost_matrix(self, current_features, tracked_features):
        """Compute cost matrix based on feature similarity."""
        cost_matrix = np.zeros((len(current_features), len(tracked_features)))
        for i, feat1 in enumerate(current_features):
            for j, feat2 in enumerate(tracked_features):
                similarity = np.dot(feat1, feat2)
                cost_matrix[i, j] = 1 - similarity
        return cost_matrix
        
    def update(self, detections, features):
        if not self.tracks:
            for i, (det, feat) in enumerate(zip(detections, features)):
                self.tracks[i] = {
                    'detection': det,
                    'feature': feat,
                    'age': 0
                }
            return list(self.tracks.keys())
            
        current_features = features
        tracked_features = [track['feature'] for track in self.tracks.values()]
        
        if not tracked_features or not current_features:
            return []
            
        cost_matrix = self.compute_cost_matrix(current_features, tracked_features)
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        
        assignments = []
        unmatched_detections = []
        matched_track_ids = set()
        
        for i, j in zip(row_ind, col_ind):
            if cost_matrix[i, j] < 1 - self.min_similarity:
                track_id = list(self.tracks.keys())[j]
                assignments.append((i, track_id))
                matched_track_ids.add(track_id)
                self.tracks[track_id] = {
                    'detection': detections[i],
                    'feature': features[i],
                    'age': 0
                }
            else:
                unmatched_detections.append(i)
        
        next_id = max(self.tracks.keys(), default=-1) + 1
        for i in unmatched_detections:
            self.tracks[next_id] = {
                'detection': detections[i],
                'feature': features[i],
                'age': 0
            }
            next_id += 1
            
        track_ids = list(self.tracks.keys())
        for track_id in track_ids:
            if track_id not in matched_track_ids:
                self.tracks[track_id]['age'] += 1
                if self.tracks[track_id]['age'] > self.max_age:
                    del self.tracks[track_id]
                    
        return [track_id for _, track_id in assignments] + list(range(next_id - len(unmatched_detections), next_id))

class FeatureExtractor:
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
    def __init__(self, camera_details_json, db_connection_string):
        camera_details = json.loads(camera_details_json)
        self.camera_urls = []
        self.camera_positions = {}
        self.door_zones = {}
        self.yolo_model = YOLO('yolov8m.pt')
        self.pose_model = YOLO('yolov8m-pose.pt')
        self.detection_confidence = 0.7
        self.feature_extractor = FeatureExtractor()
        self.db = ReIDDatabase(db_connection_string)

        self.persons = {}
        self.camera_persons = defaultdict(dict)
        self.recent_global_ids = deque(maxlen=100)
        self.local_trackers = {}
        self.match_threshold = 0.7
        self.feature_history = defaultdict(lambda: deque(maxlen=10))
        self.keypoints_history = defaultdict(lambda: defaultdict(lambda: deque(maxlen=30)))
        
        # Initialize FAISS index
        self.initialize_faiss_index()
        
        self.person_id_map = {}
        self.entry_exit_counts = defaultdict(lambda: {"entry": 0, "exit": 0})

        self._initialize_cameras(camera_details)

    def initialize_faiss_index(self):
        """Initialize CPU FAISS index."""
        dim = self.feature_extractor.feature_dim
        try:
            self.faiss_index = faiss.IndexHNSWFlat(dim, 32)
            self.faiss_index.hnsw.efConstruction = 40
            self.faiss_index.hnsw.efSearch = 16
            logging.info("Successfully initialized CPU FAISS index")
        except Exception as e:
            logging.error(f"Failed to initialize FAISS index: {str(e)}")
            raise

    def save_index(self, filepath):
        """Save the FAISS index to disk."""
        try:
            faiss.write_index(self.faiss_index, filepath)
            logging.info(f"Successfully saved FAISS index to {filepath}")
        except Exception as e:
            logging.error(f"Failed to save FAISS index: {str(e)}")
            raise

    def load_index(self, filepath):
        """Load the FAISS index from disk."""
        if not os.path.exists(filepath):
            logging.warning(f"Index file {filepath} not found")
            return False
            
        try:
            self.faiss_index = faiss.read_index(filepath)
            logging.info("Successfully loaded CPU FAISS index")
            return True
        except Exception as e:
            logging.error(f"Failed to load FAISS index: {str(e)}")
            return False

    def _initialize_cameras(self, camera_details):
        self.index_to_camera_id = {}
        for i, camera in enumerate(camera_details['cameraDetails']):
            entrance_name = camera['entranceName']
            camera_id = entrance_name
            video_url = camera['videoUrl']
            door_coords = camera['doorCoordinates']
            camera_position = camera['cameraPosition']

            self.camera_urls.append(video_url)
            self.camera_positions[camera_id] = camera_position
            self.door_zones[camera_id] = door_coords
            self.local_trackers[camera_id] = LocalTracker()
            self.index_to_camera_id[i] = camera_id

    async def initialize_state_from_db(self):
        counts = self.db.get_all_counts()
        for camera_id, count_data in counts.items():
            self.entry_exit_counts[camera_id]["entry"] = count_data.get("entry", 0)
            self.entry_exit_counts[camera_id]["exit"] = count_data.get("exit", 0)

        all_persons = self.db.get_all_persons()
        feature_batch = []
        person_ids = []
        
        for person_id, features in all_persons.items():
            person_id = int(person_id)
            features = np.array(features, dtype=np.float32)
            feature_batch.append(features)
            person_ids.append(person_id)
            self.persons[person_id] = {"features": features, "history": set()}

        if feature_batch:
            feature_batch = np.array(feature_batch, dtype=np.float32)
            self.faiss_index.add(feature_batch)
            for i, person_id in enumerate(person_ids):
                self.person_id_map[i] = person_id

    async def update_counts_in_db(self, camera_id):
        self.db.update_counts(
            camera_id,
            self.entry_exit_counts[camera_id]["entry"],
            self.entry_exit_counts[camera_id]["exit"]
        )

    def get_pose_center(self, keypoints):
        valid_keypoints = keypoints[keypoints[:, 2] > 0.5]
        if len(valid_keypoints) > 0:
            return np.mean(valid_keypoints[:, :2], axis=0)
        return None

    def calculate_movement_direction(self, current_keypoints, history_keypoints):
        if len(history_keypoints) < 2:
            return None

        current_center = self.get_pose_center(current_keypoints)
        past_center = self.get_pose_center(history_keypoints[-2])

        if current_center is not None and past_center is not None:
            movement = current_center - past_center
            if np.linalg.norm(movement) > 5:
                return movement
        return None

    def is_inside_zone(self, keypoints, door_zone):
        (x_min, y_min), (x_max, y_max) = door_zone
        center = self.get_pose_center(keypoints)
        if center is None:
            return False

        x, y = center
        margin = 20
        return (x_min - margin <= x <= x_max + margin) and (y_min - margin <= y <= y_max + margin)

    def _preprocess_image(self, person_img):
        person_img = cv2.cvtColor(person_img, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(person_img)
        transform = transforms.Compose([
            transforms.Resize((256, 128)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        return transform(pil_img).unsqueeze(0)

    async def process_frame(self, frame, camera_id):
        yolo_results = self.yolo_model.predict(frame, conf=self.detection_confidence, classes=0)
        pose_results = self.pose_model(frame, conf=self.detection_confidence)

        detections = []
        features = []
        boxes = []

        if pose_results and hasattr(pose_results[0], 'keypoints') and len(pose_results[0].keypoints) > 0:
            keypoints_data = pose_results[0].keypoints.data
            boxes_data = pose_results[0].boxes

            for i, pose in enumerate(keypoints_data):
                if i >= len(boxes_data):
                    continue

                keypoints = pose.cpu().numpy()
                box = boxes_data[i]
                x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())

                person_img = frame[y1:y2, x1:x2]
                if person_img.size == 0:
                    continue

                person_img_tensor = self._preprocess_image(person_img)
                feature = self.feature_extractor.extract(person_img_tensor)
                
                if feature is None or feature.size == 0:
                    continue
                    
                detections.append(([x1, y1, x2-x1, y2-y1], float(box.conf), 'person'))
                features.append(feature)
                boxes.append([x1, y1, x2, y2])

        if not detections:
            return frame

        # Local tracking
        local_ids = self.local_trackers[camera_id].update(detections, features)

        # Global ReID
        for local_id, (detection, feature, box) in zip(local_ids, zip(detections, features, boxes)):
            x1, y1, w, h = detection[0]
            
            # Search in FAISS
            D, I = self.faiss_index.search(np.array([feature], dtype=np.float32), 1)
            global_id = None
            
            if D[0][0] < 1 - self.match_threshold and I[0][0] < self.faiss_index.ntotal:
                global_id = self.person_id_map.get(I[0][0])
            
            if global_id is None:
                global_id = max(self.persons.keys(), default=0) + 1
                self.faiss_index.add(np.array([feature], dtype=np.float32))
                self.person_id_map[self.faiss_index.ntotal - 1] = global_id
                self.persons[global_id] = {"features": feature, "history": {camera_id}}
                self.db.store_person(global_id, feature.tolist(), camera_id)

            self.keypoints_history[camera_id][global_id].append(keypoints)
            movement = self.calculate_movement_direction(
                keypoints, list(self.keypoints_history[camera_id][global_id])
            )

            # Process transitions
            door_zone = self.door_zones[camera_id]
            position = self.camera_positions[camera_id]

            in_zone = self.is_inside_zone(keypoints, door_zone)
            was_in_zone = len(self.keypoints_history[camera_id][global_id]) > 1 and \
                          self.is_inside_zone(self.keypoints_history[camera_id][global_id][-2], door_zone)

            if movement is not None:
                if position == 'inside-out':
                    if not was_in_zone and in_zone and movement[1] > 10:
                        self.entry_exit_counts[camera_id]["exit"] += 1
                        asyncio.create_task(self.update_counts_in_db(camera_id))
                    elif was_in_zone and not in_zone and movement[1] < -10:
                        self.entry_exit_counts[camera_id]["entry"] += 1
                        asyncio.create_task(self.update_counts_in_db(camera_id))
                elif position == 'outside-in':
                    if not was_in_zone and in_zone and movement[1] < -10:
                        self.entry_exit_counts[camera_id]["entry"] += 1
                        asyncio.create_task(self.update_counts_in_db(camera_id))
                    elif was_in_zone and not in_zone and movement[1] > 10:
                        self.entry_exit_counts[camera_id]["exit"] += 1
                        asyncio.create_task(self.update_counts_in_db(camera_id))

            # Draw visualizations
            x1, y1, x2, y2 = box
            for kp in keypoints:
                if kp[2] > 0.5:
                    cv2.circle(frame, (int(kp[0]), int(kp[1])), 3, (0, 255, 0), -1)

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            # Continuing from the previous code...
            cv2.putText(frame, f"ID: {global_id}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # Draw door zone and counts
        (x_min, y_min), (x_max, y_max) = self.door_zones[camera_id]
        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 0, 255), 2)
        cv2.putText(frame, "Door Zone", (x_min, y_min - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

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
            
class IndexManager:
    """Manage HNSW index persistence and maintenance."""
    def __init__(self, index_path):
        self.index_path = index_path
        
    async def periodic_index_maintenance(self, tracker, interval_hours=24):
        """Periodically maintain the HNSW index."""
        while True:
            # Save current index state
            tracker.save_index(self.index_path)
            
            # Optimize index if needed
            if tracker.faiss_index.ntotal > 1000:
                logging.info("Performing index maintenance...")
                # Add any specific maintenance tasks here
                
            await asyncio.sleep(interval_hours * 3600)

async def main():
    # Initialize logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)
    
    tracker = None  # Initialize tracker variable in outer scope
    
    try:
        # Get database connection string from environment variables
        db_connection_string = os.getenv('COSMOS_DB_CONNECTION_STRING')
        if not db_connection_string:
            logger.error("Database connection string not found in environment variables")
            return
            
        db_handler = ReIDDatabase(db_connection_string)
        
        # Fetch camera setup details
        camera_setup_details = db_handler.get_camera_setup_details()
        if not camera_setup_details:
            logger.error("Failed to fetch camera setup details from the database")
            return
            
        camera_details_json = json.dumps({
            "cameraDetails": camera_setup_details["cameraDetails"]
        })
        
        # Initialize tracker and index manager
        tracker = MultiCameraTracker(camera_details_json, db_connection_string)
        index_manager = IndexManager("hnsw_index.faiss")
        
        # Load existing index if available
        if os.path.exists("hnsw_index.faiss"):
            logger.info("Loading existing FAISS index...")
            tracker.load_index("hnsw_index.faiss")
        
        # Create tasks for tracker and index maintenance
        tracker_task = asyncio.create_task(tracker.run())
        maintenance_task = asyncio.create_task(
            index_manager.periodic_index_maintenance(tracker)
        )
        
        # Run both tasks concurrently
        await asyncio.gather(tracker_task, maintenance_task)
        
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        raise
    finally:
        # Cleanup
        cv2.destroyAllWindows()
        if tracker:  # Check if tracker was initialized
            tracker.save_index("hnsw_index.faiss")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nShutting down gracefully...")
    except Exception as e:
        logging.error(f"Fatal error: {str(e)}")
        raise
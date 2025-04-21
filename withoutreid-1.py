import cv2
import torch
import numpy as np
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor
from collections import defaultdict, deque
import os
import logging
import json
from dotenv import load_dotenv
from db_handler1 import ReIDDatabase

class MultiCameraTracker:
    def __init__(self, camera_details_json, db_connection_string):
        camera_details = json.loads(camera_details_json)
        self.camera_urls = []
        self.camera_positions = {}
        self.door_zones = {}
        self.counting_lines = {}
        # Check for GPU availability and set device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.yolo_model = YOLO('yolov8s.pt').to(self.device)
        self.detection_confidence = 0.7
        self.db = ReIDDatabase(db_connection_string)
        self.target_resolution = (1280, 720)  # Standard resolution for better detection

        self.camera_trackers = {}
        self.track_history = defaultdict(lambda: defaultdict(lambda: deque(maxlen=30)))
        self.entry_exit_counts = defaultdict(lambda: {"entry": 0, "exit": 0})
        self.track_line_crossings = defaultdict(dict)
        self.db_connection_string = db_connection_string
        self.user_id = None
        self.camera_details = camera_details  # Store for periodic checking

        self._initialize_cameras(camera_details)
        self.url_check_task = None

    async def start_url_checker(self):
        """Start periodic camera URL checking."""
        self.url_check_task = asyncio.create_task(self._periodic_url_check())

    async def _periodic_url_check(self):
        """Check camera URLs every minute and update if changed."""
        while True:
            try:
                # Fetch the latest camera details for the user from the database
                if self.user_id is None:
                    await asyncio.sleep(60)
                    continue

                user_doc = await asyncio.to_thread(self.db.get_user_counts, self.user_id)
                if not user_doc or "cameraDetails" not in user_doc:
                    logging.warning(f"No camera details found for user {self.user_id}")
                    await asyncio.sleep(60)
                    continue

                new_camera_details = {
                    "cameraDetails": user_doc["cameraDetails"]
                }

                # Compare current and new camera details
                current_urls = {cam["videoUrl"] for cam in self.camera_details["cameraDetails"]}
                new_urls = {cam["videoUrl"] for cam in new_camera_details["cameraDetails"]}

                if current_urls != new_urls or len(self.camera_details["cameraDetails"]) != len(new_camera_details["cameraDetails"]):
                    logging.info(f"Camera details changed for user {self.user_id}. Reinitializing cameras.")
                    self.camera_details = new_camera_details
                    self._initialize_cameras(new_camera_details)
                    logging.info(f"New camera URLs: {new_urls}")
                else:
                    logging.debug(f"No changes in camera URLs for user {self.user_id}")

            except Exception as e:
                logging.error(f"Error checking camera URLs for user {self.user_id}: {str(e)}")

            await asyncio.sleep(60)  # Check every minute

    def _initialize_cameras(self, camera_details):
        """Parse camera details and initialize trackers and zones."""
        self.index_to_camera_id = {}
        self.camera_urls = []
        for i, camera in enumerate(camera_details['cameraDetails']):
            entrance_name = camera['entranceName']
            camera_id = entrance_name
            video_url = camera['videoUrl']
            door_coords = camera['doorCoordinates']
            camera_position = camera['cameraPosition']

            self.camera_urls.append(video_url)
            self.camera_positions[camera_id] = camera_position
            self.door_zones[camera_id] = door_coords
            
            (x_min, y_min), (x_max, y_max) = door_coords
            middle_y = (y_min + y_max) // 2
            self.counting_lines[camera_id] = {
                'line': ((x_min, middle_y), (x_max, middle_y)),
                'zone_height': y_max - y_min
            }

            self.camera_trackers[camera_id] = DeepSort(
                max_age=50,
                n_init=5,
                nms_max_overlap=0.6,
                max_cosine_distance=0.3,
                nn_budget=100
            )
            self.index_to_camera_id[i] = camera_id

    async def initialize_state_from_db(self, user_id):
        """Initialize counts from the database."""
        self.user_id = user_id
        user_data = await asyncio.to_thread(self.db.get_user_counts, user_id)
        
        if user_data and "cameras" in user_data:
            for camera_id, camera_data in user_data["cameras"].items():
                self.entry_exit_counts[camera_id] = {
                    "entry": camera_data.get("entry_count", 0),
                    "exit": camera_data.get("exit_count", 0)
                }
        # Start URL checker after initialization
        await self.start_url_checker()

    def is_inside_zone(self, bbox, door_zone):
        """Check if a bounding box is inside the door zone."""
        x1, y1, x2, y2 = bbox
        (zone_x_min, zone_y_min), (zone_x_max, zone_y_max) = door_zone
        
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2
        
        return (zone_x_min <= center_x <= zone_x_max) and (zone_y_min <= center_y <= zone_y_max)

    def check_line_crossing(self, point, prev_point, line):
        """Check if a point has crossed the counting line."""
        if prev_point is None:
            return None

        line_y = line[0][1]
        
        if (prev_point[1] <= line_y <= point[1]) or (point[1] <= line_y <= prev_point[1]):
            if point[1] != prev_point[1]:
                slope = (point[0] - prev_point[0]) / (point[1] - prev_point[1])
                x_cross = prev_point[0] + slope * (line_y - prev_point[1])
                
                if line[0][0] <= x_cross <= line[1][0]:
                    return "down" if prev_point[1] < point[1] else "up"
        
        return None

    def _get_detections_from_yolo(self, frame, door_zone):
        """Process YOLO detections for a frame, only within door zone."""
        results = self.yolo_model(frame, conf=self.detection_confidence, classes=0, device=self.device)
        detections = []
        
        if results and len(results[0].boxes) > 0:
            for box in results[0].boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                conf = float(box.conf)
                
                if conf < self.detection_confidence:
                    continue
                    
                if self.is_inside_zone((x1, y1, x2, y2), door_zone):
                    w, h = x2 - x1, y2 - y1
                    if w * h >= 100:
                        detections.append(([x1, y1, w, h], conf, None))
                
        return detections

    def _process_crossing(self, track_id, crossing_direction, camera_id):
        """Process line crossing and update counts."""
        if track_id not in self.track_line_crossings[camera_id]:
            self.track_line_crossings[camera_id][track_id] = crossing_direction
            
            position = self.camera_positions[camera_id]
            event_type = None
            
            if position == 'inside-out':
                if crossing_direction == "down":
                    self.entry_exit_counts[camera_id]["exit"] += 1
                    event_type = "person_exit"
                    return True, event_type
                elif crossing_direction == "up":
                    self.entry_exit_counts[camera_id]["entry"] += 1
                    event_type = "person_entry"
                    return True, event_type
            elif position == 'outside-in':
                if crossing_direction == "up":
                    self.entry_exit_counts[camera_id]["exit"] += 1
                    event_type = "person_exit"
                    return True, event_type
                elif crossing_direction == "down":
                    self.entry_exit_counts[camera_id]["entry"] += 1
                    event_type = "person_entry"
                    return True, event_type
        
        return False, None

    async def update_counts(self, camera_id, counts=None, track_id=None, event_type=None):
        """Update the database with the latest counts for the current user."""
        if self.user_id is None:
            return

        try:
            if counts is None:
                counts = self.entry_exit_counts[camera_id]
                
            entry_count = counts["entry"]
            exit_count = counts["exit"]
            
            await asyncio.to_thread(
                self.db.update_user_counts,
                self.user_id,
                camera_id,
                entry_count,
                exit_count
            )
            
            if event_type is None:
                event_type = "count_update"
                
            system_id = str(track_id) if track_id is not None else "system"
            
            await asyncio.to_thread(
                self.db.store_user_log,
                self.user_id,
                camera_id,
                system_id,
                event_type
            )
        except Exception as e:
            logging.error(f"Error updating counts for user {self.user_id}: {str(e)}")

    def _draw_visualizations(self, frame, camera_id, tracks):
        """Draw tracking visualizations on the frame."""
        (x_min, y_min), (x_max, y_max) = self.door_zones[camera_id]
        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 0, 255), 2)
        
        line = self.counting_lines[camera_id]['line']
        cv2.line(frame, line[0], line[1], (255, 0, 0), 2)
        cv2.putText(frame, "Counting Line", (line[0][0], line[0][1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

        for track in tracks:
            if not track.is_confirmed():
                continue

            ltrb = track.to_ltrb()
            x1, y1, x2, y2 = map(int, ltrb)
            
            if self.is_inside_zone((x1, y1, x2, y2), self.door_zones[camera_id]):
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"ID: {track.track_id}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        entry_count = self.entry_exit_counts[camera_id]["entry"]
        exit_count = self.entry_exit_counts[camera_id]["exit"]
        cv2.putText(frame, f"Entry: {entry_count} | Exit: {exit_count}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    async def _process_track(self, track, camera_id):
        """Process a single track and update counts if needed."""
        if not track.is_confirmed():
            return False

        track_id = track.track_id
        ltrb = track.to_ltrb()
        x1, y1, x2, y2 = map(int, ltrb)
        
        if not self.is_inside_zone((x1, y1, x2, y2), self.door_zones[camera_id]):
            return False

        center_point = ((x1 + x2) // 2, (y1 + y2) // 2)
        
        history = self.track_history[camera_id][track_id]
        prev_point = history[-1] if history else None
        history.append(center_point)
        
        crossing = self.check_line_crossing(
            center_point,
            prev_point,
            self.counting_lines[camera_id]['line']
        )
        
        if crossing:
            count_updated, event_type = self._process_crossing(track_id, crossing, camera_id)
            if count_updated:
                await self.update_counts(
                    camera_id,
                    counts=None,
                    track_id=track_id,
                    event_type=event_type
                )
                return True
        
        return False

    async def process_frame(self, frame, camera_id):
        """Process a single frame with YOLO detection and DeepSort tracking."""
        if frame is None:
            return None

        # Resize frame to target resolution
        frame = cv2.resize(frame, self.target_resolution)
        
        detections = self._get_detections_from_yolo(frame, self.door_zones[camera_id])
        tracks = self.camera_trackers[camera_id].update_tracks(detections, frame=frame)
        track_tasks = [self._process_track(track, camera_id) for track in tracks]
        await asyncio.gather(*track_tasks)
        self._draw_visualizations(frame, camera_id, tracks)
        return frame

class UserCameraProcessor:
    def __init__(self, user_id, camera_details, db_connection_string):
        self.user_id = user_id
        self.tracker = MultiCameraTracker(camera_details, db_connection_string)
        self.is_running = False
        self.thread = None
        self.db_connection_string = db_connection_string
        self.logger = logging.getLogger(__name__)

    async def start(self):
        """Start processing for this user's cameras."""
        await self.tracker.initialize_state_from_db(self.user_id)
        
        self.is_running = True
        self.thread = threading.Thread(target=self._run_tracking_loop)
        self.thread.start()

    def stop(self):
        """Stop processing for this user's cameras."""
        self.is_running = False
        if self.tracker.url_check_task:
            self.tracker.url_check_task.cancel()
        if self.thread:
            self.thread.join()

    def _run_tracking_loop(self):
        """Run the tracking loop in a separate thread."""
        asyncio.run(self._process_cameras())

    async def _process_cameras(self):
        """Process all cameras for this user."""
        caps = [cv2.VideoCapture(url) for url in self.tracker.camera_urls]
        # Set camera resolution
        for cap in caps:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.tracker.target_resolution[0])
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.tracker.target_resolution[1])
        
        camera_active = [True] * len(caps)
        
        try:
            while self.is_running and any(camera_active):
                frames = []
                for i, cap in enumerate(caps):
                    if camera_active[i]:
                        ret, frame = cap.read()
                        if not ret:
                            camera_active[i] = False
                            window_name = f"User {self.user_id} - Camera {self.tracker.index_to_camera_id[i]}"
                            cv2.destroyWindow(window_name)
                            self.logger.info(f"Video finished for {window_name}")
                            continue
                        frames.append((i, frame))
                        
                if not frames:
                    break
                    
                tasks = [
                    self.tracker.process_frame(frame, self.tracker.index_to_camera_id[i])
                    for i, frame in frames
                ]
                
                if tasks:
                    processed_frames = await asyncio.gather(*tasks)
                    
                    for i, frame in enumerate(processed_frames):
                        if frame is not None:
                            idx = frames[i][0]
                            window_name = f"User {self.user_id} - Camera {self.tracker.index_to_camera_id[idx]}"
                            cv2.imshow(window_name, frame)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                
            self.logger.info(f"All videos finished for user {self.user_id}")
            self.is_running = False
                
        except Exception as e:
            self.logger.error(f"Error processing cameras for user {self.user_id}: {str(e)}")
        finally:
            for cap in caps:
                cap.release()
            cv2.destroyAllWindows()

class MultiUserTrackingSystem:
    def __init__(self, db_connection_string):
        self.db = ReIDDatabase(db_connection_string)
        self.user_processors = {}
        self.executor = ThreadPoolExecutor(max_workers=10)
        self.logger = logging.getLogger(__name__)
        self.db_connection_string = db_connection_string
        self.user_check_task = None

    async def initialize(self):
        """Initialize the system by loading all user documents and starting user check."""
        try:
            user_documents = await self.db.get_all_user_documents()
            for doc in user_documents:
                user_id = doc["user_id"]
                camera_details = json.dumps({
                    "cameraDetails": doc["cameraDetails"]
                })
                await self.add_user_processor(user_id, camera_details)
            # Start periodic user document checker
            self.user_check_task = asyncio.create_task(self._periodic_user_check())
        except Exception as e:
            self.logger.error(f"Failed to initialize system: {str(e)}")
            raise

    async def _periodic_user_check(self):
        """Check for new user documents every minute."""
        while True:
            try:
                # Fetch all user documents
                user_documents = await self.db.get_all_user_documents()
                current_user_ids = set(self.user_processors.keys())
                new_user_ids = {doc["user_id"] for doc in user_documents}

                # Check for new users
                for doc in user_documents:
                    user_id = doc["user_id"]
                    if user_id not in current_user_ids:
                        self.logger.info(f"New user document found: {user_id}")
                        camera_details = json.dumps({
                            "cameraDetails": doc["cameraDetails"]
                        })
                        await self.add_user_processor(user_id, camera_details)

                # Check for deleted users (optional)
                for user_id in current_user_ids:
                    if user_id not in new_user_ids:
                        self.logger.info(f"User document removed: {user_id}")
                        await self.stop_user_processor(user_id)

            except Exception as e:
                self.logger.error(f"Error checking for new user documents: {str(e)}")

            await asyncio.sleep(60)  # Check every minute

    async def add_user_processor(self, user_id, camera_details):
        """Add or update a user processor."""
        if user_id in self.user_processors:
            await self.stop_user_processor(user_id)
        
        processor = UserCameraProcessor(user_id, camera_details, self.db_connection_string)
        self.user_processors[user_id] = processor
        await processor.start()

    async def stop_user_processor(self, user_id):
        """Stop processing for a specific user."""
        if user_id in self.user_processors:
            self.user_processors[user_id].stop()
            del self.user_processors[user_id]

    async def stop_all(self):
        """Stop all user processors and clean up."""
        if self.user_check_task:
            self.user_check_task.cancel()
        for user_id in list(self.user_processors.keys()):
            await self.stop_user_processor(user_id)
        self.executor.shutdown()
        cv2.destroyAllWindows()

    async def get_user_statistics(self, user_id):
        """Get statistics for a specific user."""
        try:
            counts = await asyncio.to_thread(self.db.get_user_total_counts, user_id)
            logs = await asyncio.to_thread(self.db.get_user_logs, user_id, 50)
            
            recent_activity = {}
            if logs:
                for log in logs:
                    camera_id = log.get("camera_id")
                    event_type = log.get("event_type")
                    
                    if camera_id not in recent_activity:
                        recent_activity[camera_id] = {
                            "person_entry": 0,
                            "person_exit": 0,
                            "count_update": 0
                        }
                    
                    if event_type in recent_activity[camera_id]:
                        recent_activity[camera_id][event_type] += 1
            
            return {
                "counts": counts,
                "recent_activity": recent_activity,
                "recent_logs": logs[:10]
            }
        except Exception as e:
            self.logger.error(f"Failed to get statistics for user {user_id}: {str(e)}")
            return {
                "counts": {"total_entry": 0, "total_exit": 0, "camera_count": 0},
                "recent_activity": {},
                "recent_logs": []
            }

async def main():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('multi_user_tracking.log'),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)
    
    try:
        load_dotenv()
        db_connection_string = os.getenv('COSMOS_DB_CONNECTION_STRING')
        if not db_connection_string:
            raise ValueError("Database connection string not found in environment variables")
        
        tracking_system = MultiUserTrackingSystem(db_connection_string)
        await tracking_system.initialize()
        
        try:
            while True:
                await asyncio.sleep(1)
        except KeyboardInterrupt:
            logger.info("Shutting down...")
            await tracking_system.stop_all()
            
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        raise

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nShutting down gracefully...")
    except Exception as e:
        logging.error(f"Fatal error: {str(e)}")
        raise
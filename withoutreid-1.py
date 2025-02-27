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
from datetime import datetime, timezone
from dotenv import load_dotenv
from db_handler1 import ReIDDatabase

class MultiCameraTracker:
    def __init__(self, camera_details_json, db_connection_string):
        camera_details = json.loads(camera_details_json)
        self.camera_urls = []
        self.camera_positions = {}
        self.door_zones = {}
        self.counting_lines = {}
        self.yolo_model = YOLO('yolov8s.pt')
        self.detection_confidence = 0.7
        self.db = ReIDDatabase(db_connection_string)

        self.camera_trackers = {}
        self.track_history = defaultdict(lambda: defaultdict(lambda: deque(maxlen=30)))   
        self.entry_exit_counts = defaultdict(lambda: {"entry": 0, "exit": 0})
        self.track_line_crossings = defaultdict(dict)
        self.db_connection_string = db_connection_string
        self.user_id = None  # Store user_id for database updates

        self._initialize_cameras(camera_details)

    def _initialize_cameras(self, camera_details):
        """Parse camera details and initialize trackers and zones."""
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
        self.user_id = user_id  # Store user_id for later use
        counts = self.db.get_user_counts(user_id)
        
        # If counts is a list, convert it to a dictionary format
        if isinstance(counts, list):
            for count_entry in counts:
                camera_id = count_entry.get("camera_id")
                if camera_id:
                    self.entry_exit_counts[camera_id]["entry"] = count_entry.get("entry_count", 0)
                    self.entry_exit_counts[camera_id]["exit"] = count_entry.get("exit_count", 0)
        else:
            # Original code assuming counts is a dictionary
            for camera_id, count_data in counts.items():
                self.entry_exit_counts[camera_id]["entry"] = count_data.get("entry", 0)
                self.entry_exit_counts[camera_id]["exit"] = count_data.get("exit", 0)

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
        results = self.yolo_model(frame, conf=self.detection_confidence, classes=0)
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
            if position == 'inside-out':
                if crossing_direction == "down":
                    self.entry_exit_counts[camera_id]["exit"] += 1
                    return True
                elif crossing_direction == "up":
                    self.entry_exit_counts[camera_id]["entry"] += 1
                    return True
            elif position == 'outside-in':
                if crossing_direction == "up":
                    self.entry_exit_counts[camera_id]["exit"] += 1
                    return True
                elif crossing_direction == "down":
                    self.entry_exit_counts[camera_id]["entry"] += 1
                    return True
        
        return False

    async def update_counts(self, camera_id, counts=None):
        """Update the database with the latest counts for the current user.
        This is the centralized method for all count updates."""
        if self.user_id is None:
            return

        try:
            # If no counts are provided, use the ones from memory
            if counts is None:
                counts = self.entry_exit_counts[camera_id]
                    
            document = {
                "user_id": self.user_id,
                "camera_id": camera_id,
                "entry_count": counts["entry"],
                "exit_count": counts["exit"],
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
            await asyncio.to_thread(self.db.update_user_counts, document)
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
        
        if crossing and self._process_crossing(track_id, crossing, camera_id):
            # Use the centralized update method
            await self.update_counts(camera_id)
            return True
        
        return False

    async def process_frame(self, frame, camera_id):
        """Process a single frame with YOLO detection and DeepSort tracking."""
        if frame is None:
            return None

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
        self.is_running = True
        self.thread = threading.Thread(target=self._run_tracking_loop)
        self.thread.start()

    def stop(self):
        """Stop processing for this user's cameras."""
        self.is_running = False
        if self.thread:
            self.thread.join()

    def _run_tracking_loop(self):
        """Run the tracking loop in a separate thread."""
        asyncio.run(self._process_cameras())

    async def _process_cameras(self):
        """Process all cameras for this user."""
        await self.tracker.initialize_state_from_db(self.user_id)
        caps = [cv2.VideoCapture(url) for url in self.tracker.camera_urls]
        camera_active = [True] * len(caps)  # Track which cameras are still active
        
        try:
            while self.is_running and any(camera_active):
                # Collect frames with status check
                frames = []
                for i, cap in enumerate(caps):
                    if camera_active[i]:
                        ret, frame = cap.read()
                        if not ret:  # Video has ended
                            camera_active[i] = False
                            window_name = f"User {self.user_id} - Camera {self.tracker.index_to_camera_id[i]}"
                            cv2.destroyWindow(window_name)
                            self.logger.info(f"Video finished for {window_name}")
                            continue
                        frames.append((i, frame))
                        
                if not frames:  # No active frames left
                    break
                    
                # Process available frames
                tasks = [
                    self.tracker.process_frame(frame, self.tracker.index_to_camera_id[i])
                    for i, frame in frames
                ]
                
                if tasks:  # Only process if we have tasks
                    processed_frames = await asyncio.gather(*tasks)
                    
                    # Update counts with user ID for each camera
                    for camera_id in self.tracker.entry_exit_counts:
                        await self.tracker.update_counts(camera_id)
                    
                    # Display frames
                    for i, frame in enumerate(processed_frames):
                        if frame is not None:
                            idx = frames[i][0]  # Get original index
                            window_name = f"User {self.user_id} - Camera {self.tracker.index_to_camera_id[idx]}"
                            cv2.imshow(window_name, frame)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                
            # If we're here, all videos are finished
            self.logger.info(f"All videos finished for user {self.user_id}")
            self.is_running = False
                
        except Exception as e: 
            self.logger.error(f"Error processing cameras for user {self.user_id}: {str(e)}")
        finally:
            for cap in caps:
                cap.release()
            # Ensure all windows are closed
            cv2.destroyAllWindows()

class MultiUserTrackingSystem:
    def __init__(self, db_connection_string):
        self.db = ReIDDatabase(db_connection_string)
        self.user_processors = {}
        self.executor = ThreadPoolExecutor(max_workers=10)
        self.logger = logging.getLogger(__name__)
        self.db_connection_string = db_connection_string

    async def initialize(self):
        """Initialize the system by loading all user documents."""
        try:
            user_documents = await self.db.get_all_user_documents()
            for doc in user_documents:
                user_id = doc["user_id"]
                camera_details = json.dumps({
                    "cameraDetails": doc["cameraDetails"]
                })
                await self.add_user_processor(user_id, camera_details)
        except Exception as e:
            self.logger.error(f"Failed to initialize system: {str(e)}")
            raise

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
        for user_id in list(self.user_processors.keys()):
            await self.stop_user_processor(user_id)
        self.executor.shutdown()
        cv2.destroyAllWindows()

async def main():
    # Initialize logging
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
        # Load environment variables
        load_dotenv()
        db_connection_string = os.getenv('COSMOS_DB_CONNECTION_STRING')
        if not db_connection_string:
            raise ValueError("Database connection string not found in environment variables")
        
        # Initialize the multi-user tracking system
        tracking_system = MultiUserTrackingSystem(db_connection_string)
        await tracking_system.initialize()
        
        # Keep the main program running
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
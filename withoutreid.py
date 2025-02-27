import cv2
import torch
import numpy as np
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import asyncio
from collections import defaultdict, deque
import os
import logging
import json
from dotenv import load_dotenv
from db_handler1 import ReIDDatabase

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
load_dotenv()

class MultiCameraTracker:
    def __init__(self, camera_details_json, db_connection_string):
        camera_details = json.loads(camera_details_json)
        self.camera_urls = []
        self.camera_positions = {}
        self.door_zones = {}
        self.counting_lines = {}  # Store middle lines for counting
        self.yolo_model = YOLO('yolov8s.pt')
        self.detection_confidence = 0.7
        self.db = ReIDDatabase(db_connection_string)

        self.camera_trackers = {}
        self.track_history = defaultdict(lambda: defaultdict(lambda: deque(maxlen=30)))
        self.entry_exit_counts = defaultdict(lambda: {"entry": 0, "exit": 0})
        self.track_line_crossings = defaultdict(dict)  # Track which IDs have crossed the line
        self.db_connection_string = db_connection_string

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
            
            # Calculate middle line for counting
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

    async def initialize_state_from_db(self):
        """Initialize counts from the database."""
        counts = self.db.get_all_counts()
        for camera_id, count_data in counts.items():
            self.entry_exit_counts[camera_id]["entry"] = count_data.get("entry", 0)
            self.entry_exit_counts[camera_id]["exit"] = count_data.get("exit", 0)

    async def update_counts_in_db(self, camera_id):
        """Update counts in the database for a specific camera."""
        self.db.update_counts(
            camera_id,
            self.entry_exit_counts[camera_id]["entry"],
            self.entry_exit_counts[camera_id]["exit"]
        )

    def is_inside_zone(self, bbox, door_zone):
        """Check if a bounding box is inside the door zone."""
        x1, y1, x2, y2 = bbox
        (zone_x_min, zone_y_min), (zone_x_max, zone_y_max) = door_zone
        
        # Calculate the center of the bounding box
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2
        
        # Check if the center point is inside the door zone
        return (zone_x_min <= center_x <= zone_x_max) and (zone_y_min <= center_y <= zone_y_max)

    def check_line_crossing(self, point, prev_point, line):
        """Check if a point has crossed the counting line."""
        if prev_point is None:
            return None

        line_y = line[0][1]  # Y-coordinate of the horizontal line
        
        # Check if the point crossed the line from either direction
        if (prev_point[1] <= line_y <= point[1]) or (point[1] <= line_y <= prev_point[1]):
            # Calculate x-coordinate where crossing occurred
            if point[1] != prev_point[1]:  # Avoid division by zero
                slope = (point[0] - prev_point[0]) / (point[1] - prev_point[1])
                x_cross = prev_point[0] + slope * (line_y - prev_point[1])
                
                # Check if crossing point is within line segment
                if line[0][0] <= x_cross <= line[1][0]:
                    # Determine direction of crossing
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
                    
                # Only include detections inside the door zone
                if self.is_inside_zone((x1, y1, x2, y2), door_zone):
                    w, h = x2 - x1, y2 - y1
                    if w * h >= 100:  # Filter out small detections
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
            elif position == 'outside-in':  # outside-in
                if crossing_direction == "up":
                    self.entry_exit_counts[camera_id]["exit"] += 1
                    return True
                elif crossing_direction == "down":
                    self.entry_exit_counts[camera_id]["entry"] += 1
                    return True
        
        return False

    def _draw_visualizations(self, frame, camera_id, tracks):
        """Draw tracking visualizations on the frame."""
        # Draw door zone
        (x_min, y_min), (x_max, y_max) = self.door_zones[camera_id]
        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 0, 255), 2)
        
        # Draw counting line
        line = self.counting_lines[camera_id]['line']
        cv2.line(frame, line[0], line[1], (255, 0, 0), 2)
        cv2.putText(frame, "Counting Line", (line[0][0], line[0][1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

        # Draw tracks
        for track in tracks:
            if not track.is_confirmed():
                continue

            ltrb = track.to_ltrb()
            x1, y1, x2, y2 = map(int, ltrb)
            
            # Only draw tracks inside door zone
            if self.is_inside_zone((x1, y1, x2, y2), self.door_zones[camera_id]):
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"ID: {track.track_id}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # Draw counts
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
        
        # Only process tracks inside door zone
        if not self.is_inside_zone((x1, y1, x2, y2), self.door_zones[camera_id]):
            return False

        center_point = ((x1 + x2) // 2, (y1 + y2) // 2)
        
        # Update track history
        history = self.track_history[camera_id][track_id]
        prev_point = history[-1] if history else None
        history.append(center_point)
        
        # Check for line crossing
        crossing = self.check_line_crossing(
            center_point, 
            prev_point, 
            self.counting_lines[camera_id]['line']
        )
        
        if crossing and self._process_crossing(track_id, crossing, camera_id):
                await self.update_counts_in_db(camera_id)
                return True
        
        return False

    async def process_frame(self, frame, camera_id):
        """Process a single frame with YOLO detection and DeepSort tracking.""" 
        if frame is None:
            return None

        # Get YOLO detections only within door zone
        detections = self._get_detections_from_yolo(frame, self.door_zones[camera_id])
        
        # Update trackers
        tracks = self.camera_trackers[camera_id].update_tracks(detections, frame=frame)

        # Process each track
        track_tasks = [self._process_track(track, camera_id) for track in tracks]
        await asyncio.gather(*track_tasks)

        # Draw visualizations
        self._draw_visualizations(frame, camera_id, tracks)

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

async def main():
    # Initialize logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)
    
    try:
        db_connection_string = os.getenv('COSMOS_DB_CONNECTION_STRING')
        db_handler = ReIDDatabase(db_connection_string)
        
        # Fetch camera setup details from database
        camera_setup_details = db_handler.get_camera_setup_details()
        if not camera_setup_details:
            logger.error("Failed to fetch camera setup details from the database. Exiting...")
            return
            
        camera_details_json = json.dumps({
            "cameraDetails": camera_setup_details["cameraDetails"]
        })
        
        # Initialize and run tracker
        tracker = MultiCameraTracker(camera_details_json, db_connection_string)
        await tracker.run()
        
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        raise
    finally:
        cv2.destroyAllWindows()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nShutting down gracefully...")
    except Exception as e:
        logging.error(f"Fatal error: {str(e)}")
        raise
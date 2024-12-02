import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import logging
import asyncio
from datetime import datetime
from pathlib import Path
import threading
from queue import Queue
from db_handler import ReIDDatabase
import torchreid

class PersonReID:
    def __init__(self, yolo_path, connection_string,  model_path):
        self.setup_logging()
        self.setup_models(yolo_path,model_path)
        self.setup_database(connection_string)
        self.setup_tracking()
        self.frame_processors = {}
        self.processing_queue = Queue()
        self.running = True
        
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((256, 128), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    def setup_logging(self):
        """Configure logging"""
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        handler = logging.FileHandler('reid_system.log')
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)

    def setup_models(self, yolo_path, model_path):
        """Initialize YOLO and Market-1501 OSNet models"""
        try:
            # Initialize YOLO
            self.yolo = YOLO(yolo_path)
            
            # Initialize OSNet with Market-1501 configuration
            self.reid_model = torchreid.models.build_model(
                name='osnet_x1_0',
                num_classes=751,  # Market-1501 has 751 identities
                loss='softmax',
                pretrained=False  # We'll load our own weights
            )
            
            # Load the specific Market-1501 weights
            checkpoint = torch.load(model_path, map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
            
            # Handle different checkpoint formats
            if isinstance(checkpoint, dict):
                if 'state_dict' in checkpoint:
                    state_dict = checkpoint['state_dict']
                else:
                    state_dict = checkpoint  # The checkpoint itself might be the state dict
            else:
                state_dict = checkpoint
                
            # Remove 'module.' prefix if it exists (happens when model was trained with DataParallel)
            from collections import OrderedDict
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                name = k.replace('module.', '') if k.startswith('module.') else k
                new_state_dict[name] = v
                
            # Load the state dict
            self.reid_model.load_state_dict(new_state_dict, strict=False)
            
            if torch.cuda.is_available():
                self.reid_model = self.reid_model.cuda()
            self.reid_model.eval()
            
            self.logger.info("Models initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize models: {str(e)}")
            raise

    def setup_database(self, connection_string):
        """Initialize database connection"""
        try:
            self.db = ReIDDatabase(connection_string)
            self.logger.info("Database connection established")
        except Exception as e:
            self.logger.error(f"Failed to connect to database: {str(e)}")
            raise

    def setup_tracking(self):
        """Initialize DeepSort tracker"""
        self.trackers = {}  # Separate tracker for each camera
        self.track_history = {}  # Store track history for each camera
        
        # DeepSort parameters
        max_age = 30  # Maximum number of frames to keep track alive
        n_init = 3    # Number of frames needed to confirm a track
        max_iou_distance = 0.7
        max_cosine_distance = 0.3
        
        # Initialize a tracker for each camera (will be done dynamically)
        self.tracker_params = {
            "max_age": max_age,
            "n_init": n_init,
            "max_iou_distance": max_iou_distance,
            "max_cosine_distance": max_cosine_distance,
            "nn_budget": 100
        }

    def preprocess_detection(self, frame, bbox):
        """Extract and preprocess person crop from frame"""
        try:
            x1, y1, x2, y2 = map(int, bbox)
            person_crop = frame[y1:y2, x1:x2]
            if person_crop.size == 0:
                return None
            
            # Add padding to make the crop more suitable for ReID
            height, width = person_crop.shape[:2]
            aspect_ratio = width / height
            
            if aspect_ratio < 0.5:  # Too tall
                pad_width = int(height * 0.5) - width
                person_crop = cv2.copyMakeBorder(
                    person_crop, 0, 0, pad_width//2, pad_width//2,
                    cv2.BORDER_CONSTANT, value=(0, 0, 0)
                )
            
            # Preprocess for OSNet model using PyTorch transforms
            person_tensor = self.transform(person_crop)
            if torch.cuda.is_available():
                person_tensor = person_tensor.cuda()
            person_tensor = person_tensor.unsqueeze(0)  # Add batch dimension
            
            return person_tensor
        except Exception as e:
            self.logger.error(f"Failed to preprocess detection: {str(e)}")
            return None

    def compute_similarity(self, features1, features2):
        """Compute cosine similarity between two feature vectors"""
        return np.dot(features1, features2) / (
            np.linalg.norm(features1) * np.linalg.norm(features2)
        )

    async def process_frame(self, camera_id, frame):
        """Process a single frame from a camera, detecting and tracking multiple persons"""
        try:
            # YOLOv8 detection for all persons
            results = self.yolo(frame, classes=0)  # class 0 is person
            detections = results[0].boxes.data.cpu().numpy()
            
            # Initialize tracker if needed
            if camera_id not in self.trackers:
                self.trackers[camera_id] = DeepSort(**self.tracker_params)
                self.track_history[camera_id] = {}
            
            # Prepare detections for DeepSort
            detection_list = []
            for det in detections:
                x1, y1, x2, y2, conf, _ = det
                detection_list.append(([x1, y1, x2 - x1, y2 - y1], conf, None))
            
            # Update tracks
            tracks = self.trackers[camera_id].update_tracks(detection_list, frame=frame)
            
            # Process each track
            processed_tracks = {}
            for track in tracks:
                if not track.is_confirmed():
                    continue
                    
                track_id = track.track_id
                
                # Get bounding box coordinates
                bbox = track.to_ltrb()
                
                # Draw bounding box
                self.visualize_detection(frame, bbox, track_id)
                
                # Extract and process person crop
                person_tensor = self.preprocess_detection(frame, bbox)
                if person_tensor is None:
                    continue

                # Extract ReID features using OSNet
                with torch.no_grad():
                    features = self.reid_model(person_tensor)
                    features = features.cpu().numpy().flatten()
                    # L2 normalization
                    features = features / np.linalg.norm(features)

                # Store features for this specific frame and track
                processed_tracks[track_id] = {
                    'bbox': bbox,
                    'features': features
                }

                # Update database with track features
                await self.db.update_person_features(str(track_id), features, camera_id)

            # Cross-track feature matching within the same frame
            track_ids = list(processed_tracks.keys())
            for i in range(len(track_ids)):
                for j in range(i+1, len(track_ids)):
                    id1, id2 = track_ids[i], track_ids[j]
                    
                    # Compute similarity between tracks
                    similarity = self.compute_similarity(
                        processed_tracks[id1]['features'], 
                        processed_tracks[id2]['features']
                    )
                    
                    # Check if tracks are similar
                    if similarity > 0.85:  # Matching threshold
                        self.logger.info(f"Potential match found between Track {id1} and Track {id2}")
                        
                        # Optional: Visualize matching with different color or annotation
                        self.visualize_match(
                            frame, 
                            processed_tracks[id1]['bbox'], 
                            processed_tracks[id2]['bbox']
                        )

            return frame

        except Exception as e:
            self.logger.error(f"Error processing frame from camera {camera_id}: {str(e)}")
            return frame

    def visualize_match(self, frame, bbox1, bbox2):
        """Highlight potential matching tracks"""
        try:
            # Convert bounding boxes to integers
            x1, y1, x2, y2 = map(int, bbox1)
            x3, y3, x4, y4 = map(int, bbox2)
            
            # Draw matching tracks in red
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
            cv2.rectangle(frame, (x3, y3), (x4, y4), (0, 0, 255), 3)
            
            # Draw a line connecting the centers of matched tracks
            center1 = ((x1 + x2) // 2, (y1 + y2) // 2)
            center2 = ((x3 + x4) // 2, (y3 + y4) // 2)
            cv2.line(frame, center1, center2, (0, 0, 255), 2)
        except Exception as e:
            self.logger.error(f"Error visualizing match: {str(e)}")

    class CameraProcessor(threading.Thread):
        """Thread class for processing camera feeds"""
        def __init__(self, camera_id, queue, source):
            super().__init__()
            self.camera_id = camera_id
            self.queue = queue
            self.source = source
            self.running = True

        def run(self):
            cap = cv2.VideoCapture(self.source)
            while self.running:
                ret, frame = cap.read()
                if ret:
                    self.queue.put((self.camera_id, frame))
            cap.release()

        def stop(self):
            self.running = False

    async def run(self, camera_sources):
        """Main loop for running the ReID system"""
        try:
            # Initialize camera processors
            for camera_id, source in camera_sources.items():
                processor = self.CameraProcessor(camera_id, self.processing_queue, source)
                self.frame_processors[camera_id] = processor
                processor.start()

            while self.running:
                if not self.processing_queue.empty():
                    camera_id, frame = self.processing_queue.get()
                    processed_frame = await self.process_frame(camera_id, frame)
                    
                    # Ensure frame is not None before displaying
                    if processed_frame is not None:
                        cv2.imshow(f"Camera {camera_id}", processed_frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        except Exception as e:
            self.logger.error(f"Error in main loop: {str(e)}")
        finally:
            self.cleanup()

    def cleanup(self):
        """Cleanup resources"""
        self.running = False
        for processor in self.frame_processors.values():
            processor.stop()
        cv2.destroyAllWindows()
        self.logger.info("System shutdown complete")

async def main():
    # Configuration
    yolo_path = "yolov8n.pt"  # or your custom trained model
    connection_string = "AccountEndpoint=https://occupancytrackerdb.documents.azure.com:443/;AccountKey=NTTvzWNTTmZ3I0rydqqnIIjPDGG5RxXVCYa9WS78XK4PvUXUGCS9Tx9s8xnfs4rSfS2xD2deHAGUACDbIMdVxA==;"
    model_path = "osnet_x1_0_market_256x128_amsgrad_ep150_stp60_lr0.0015_b64_fb10_softmax_labelsmooth_flip.pth"
    camera_sources = {
        0: 0,  # Camera 1
        1: r'rtsp://Yectra:Yectra123@192.168.1.15:554/stream2',  # Camera 2
        # Add more cameras as needed
    }

    # Initialize and run system
    reid_system = PersonReID(yolo_path, connection_string,model_path)
    await reid_system.run(camera_sources)

if __name__ == "__main__":
    asyncio.run(main())
import cv2
import numpy as np
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import argparse
from collections import defaultdict
import torch
import os

class BarbellTracker:
    """
    Class for detecting and tracking barbells in videos using YOLOv8 and DeepSORT.
    Pure tracking implementation with no visualization components.
    """
    def __init__(self, model_path, confidence=0.5, device='cuda'):
        self.confidence = confidence
        self.device = device
        
        # Initialize YOLO model
        print(f"Loading model from {model_path} on {device}...")
        self.model = YOLO(model_path)
        self.model.to(device)

        # Initialize DeepSORT tracker
        print("Initializing DeepSORT tracker...")
        self.tracker = DeepSort(
            max_age=30,
            nn_budget=100,
            embedder="mobilenet",
            embedder_gpu=(device == 'cuda')
        )
        
        # Dictionary to store trajectories
        self.trajectories = defaultdict(list)
        
    def get_video_properties(self, video_path):
        """Get video properties (width, height, fps)"""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Unable to open video {video_path}")
            return None, None, None
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        return width, height, fps

    def process_video(self, video_path):
        """
        Process video and track barbell trajectory.
        Pure tracking with no visualization.
        
        Args:
            video_path (str): Path to input video
            
        Returns:
            dict: Dictionary mapping track IDs to trajectories
        """
        # Reset trajectories
        self.trajectories = defaultdict(list)
        
        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Unable to open video {video_path}")
            return None
        
        # Get video properties
        _, _, fps = self.get_video_properties(video_path)
        if fps is None:
            return None
        
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            timestamp = frame_count / fps  # Time in seconds
            
            if frame_count % 30 == 0:  # Print progress every 30 frames
                print(f"Processing frame {frame_count}")
    
            # Run detection
            results = self.model(frame, conf=self.confidence, verbose=False)
            
            # Extract bounding boxes for barbells
            detections = []
            
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    # Check confidence
                    conf = float(box.conf[0])
                    cls = int(box.cls[0])
                    
                    if conf > self.confidence:
                        # Get bounding box coordinates
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        
                        # Format for DeepSORT: (bbox, confidence, class)
                        # bbox format for DeepSORT: [x, y, width, height]
                        detections.append(([x1, y1, x2-x1, y2-y1], conf, cls))
            
            # Update DeepSORT tracker
            tracks = self.tracker.update_tracks(detections, frame=frame)
            
            # Update trajectories - no visualization, just data collection
            for track in tracks:
                if not track.is_confirmed():
                    continue
                
                track_id = track.track_id
                ltrb = track.to_ltrb()
                
                x1, y1, x2, y2 = int(ltrb[0]), int(ltrb[1]), int(ltrb[2]), int(ltrb[3])
                
                # Calculate center point
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2
                
                # Add point to trajectory with timestamp
                self.trajectories[track_id].append((center_x, center_y, timestamp))
        
        # Clean up
        cap.release()
        
        print(f"Processed {frame_count} frames, found {len(self.trajectories)} tracked objects")
        
        # Return trajectories
        return {k: v for k, v in self.trajectories.items()}
    
    def get_longest_trajectory(self):
        """
        Get the longest trajectory from the tracking results.
        Usually corresponds to the consistently tracked barbell.
        
        Returns:
            tuple: (track_id, trajectory_points) or (None, None) if no trajectories
        """
        if not self.trajectories:
            return None, None
            
        # Find the longest trajectory
        longest_track_id = None
        max_length = 0
        
        for track_id, trajectory in self.trajectories.items():
            if len(trajectory) > max_length:
                max_length = len(trajectory)
                longest_track_id = track_id
                
        if longest_track_id is None:
            return None, None
            
        return longest_track_id, self.trajectories[longest_track_id]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Track barbell trajectory in a video.')
    parser.add_argument('--input', required=True, help='Input video path')
    parser.add_argument('--model', required=True, help='Path to YOLOv8 model')
    parser.add_argument('--confidence', type=float, default=0.5, help='Detection confidence threshold (default: 0.5)')
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'], help='Device to use (default: cuda)')
    
    args = parser.parse_args()
    
    # Check CUDA availability if requested
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("Warning: CUDA requested but not available. Falling back to CPU.")
        args.device = 'cpu'
    else:
        print(f"Using device: {args.device}")
        if args.device == 'cuda':
            print(f"CUDA Device: {torch.cuda.get_device_name(0)}")
    
    # Initialize and run the tracker
    tracker = BarbellTracker(args.model, args.confidence, args.device)
    trajectories = tracker.process_video(args.input)
    
    if trajectories:
        print(f"Found {len(trajectories)} tracked objects")
        track_id, trajectory = tracker.get_longest_trajectory()
        if track_id is not None:
            print(f"Longest trajectory has ID {track_id} with {len(trajectory)} points")
            print(f"First 3 points: {trajectory[:3]}")
            print(f"Last 3 points: {trajectory[-3:]}")
    else:
        print("No trajectories were detected")
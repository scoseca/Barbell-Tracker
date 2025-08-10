import argparse
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from ultralytics import YOLO
# Correzione del percorso di importazione (scr → src)
import src.detection.barbell_tracker as barbell_tracker_module


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='main')
    parser.add_argument('--yolo_model_path', type=str, required=True, help='YOLOv8 model path')
    parser.add_argument('--pose_landmarker_path', type=str, required=True, help='Pose Landmarker model path')
    parser.add_argument('--input', type=str, required=True, help='video path')
    parser.add_argument('--confidence', type=float, default=0.5, help='Detection confidence threshold (default: 0.5)')
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'], help='Device to use (default: cuda)')

    args = parser.parse_args()
    
    # Configura MediaPipe Pose Landmarker
    base_options = python.BaseOptions(model_asset_path=args.pose_landmarker_path)
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        output_segmentation_masks=False,
        min_pose_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        min_pose_presence_confidence=0.5,
        num_poses=1)
    
    pose_landmarker = vision.PoseLandmarker.create_from_options(options)
    
    # Inizializza il tracker del bilanciere (evitare doppia inizializzazione del modello YOLO)
    tracker = barbell_tracker_module.BarbellTracker(
        model_path=args.yolo_model_path,
        confidence=args.confidence, 
        device=args.device
    )
    
    # Esegui il tracking sul video
    trajectories = tracker.process_video(args.input)
    
    # Analizza i risultati
    if trajectories:
        print(f"Found {len(trajectories)} tracked objects")
        track_id, trajectory = tracker.get_longest_trajectory()
        if track_id is not None:
            print(f"Longest trajectory has ID {track_id} with {len(trajectory)} points")
            print(f"First 3 points: {trajectory[:3]}")
            print(f"Last 3 points: {trajectory[-3:]}")
    else:
        print("No trajectories were detected")
    
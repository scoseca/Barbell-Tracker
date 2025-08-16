import argparse
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from ultralytics import YOLO
import src.detection.barbell_tracker as barbell_tracker_module
from src.analysis.trajectory_analysis import TrajectoryAnalyzer
from src.visualization.visualizer import Visualizer

def print_analysis_results(results_dict):
    """
    Formatta e stampa i risultati dell'analisi della traiettoria.
    """
    print("\n----- ANALISI DELLA TRAIETTORIA -----")
    for key, value in results_dict.items():
        print(f"{key}:")
        
        if isinstance(value, (int, float)):
            print(f"  {value:.2f}")
        elif isinstance(value, np.ndarray) and key == "positions":
            print("  [")
            for i, pos in enumerate(value):
                #if i % 10 == 0:  # Stampa ogni 10 posizioni (puoi regolarlo)
                print(f"    Punto {i}: ({pos[0]:.2f}, {pos[1]:.2f})")
            #print(f"    ... e altri {len(value)-4} punti")
            print("  ]")
        elif isinstance(value, np.ndarray):
            if value.ndim == 1:
                print(f"  Array 1D di lunghezza {len(value)}")
            else:
                print(f"  Array {value.ndim}D di forma {value.shape}")
        elif isinstance(value, list):
            print(f"  Lista di lunghezza {len(value)}")
        else:
            print(f"  {value}")
    print("-------------------------------------\n")


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
    
    # Inizializza il tracker del bilanciere
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
            # print(f"Longest trajectory has ID {track_id} with {len(trajectory)} points")
            # print(f"First 3 points: {trajectory[:3]}")
            # print(f"Last 3 points: {trajectory[-3:]}")
            
            # Inizializza l'analizzatore con l'fps corretto
            # Ottieni l'fps dal tracker o dal video
            fps = 30  # Sostituisci con l'fps reale del tuo video
            analyzer = TrajectoryAnalyzer(fps=fps)
            
            # Da [(x, y, timestamp), ...] a [(x, y), ...]
            # La TrajectoryAnalyzer probabilmente si aspetta solo coordinate x,y senza timestamp
            trajectory_xy = [(point[0], point[1]) for point in trajectory]
            
            # Ora passa solo le coordinate x,y all'analyzer
            smooth_trajectory = analyzer.smooth_trajectory(trajectory_xy)
            if trajectory_xy is not None:
                analysis_results = analyzer.analyze_trajectory(trajectory_xy)
                print_analysis_results(analysis_results)
    else:
        print("No trajectories were detected")

    visualizer = Visualizer()
    # visualizer.plot_trajectory_2d(smooth_trajectory, save_path="E:/Tesi/Barbel-Tracker/data/output")
    # visualizer.plot_metrics(analysis_results)
    # try:
    #     print("Creazione dell'animazione...")
    #     trajectory_array = np.array(smooth_trajectory)
    #     visualizer.create_animation(
    #         trajectory_array,
    #         analysis_results,
    #         title="Analisi della Traiettoria del Bilanciere",
    #         save_path="barbell_trajectory_animation.gif",
    #         fps=10
    #     )
    #     print("Animazione creata con successo!")
    # except Exception as e:
    #     print(f"Errore durante la creazione dell'animazione: {str(e)}")

    visualizer.plot_trajectory_on_video(smooth_trajectory, 
                                    video_path=args.input,
                                    output_path="E:/Tesi/Barbel-Tracker/data/output/trajectory_video.mp4",
                                    show_velocity=True,
                                    analysis_results=analysis_results)

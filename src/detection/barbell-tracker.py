import cv2
import numpy as np
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import argparse
from collections import defaultdict
import torch
import os
from datetime import datetime
from src.analysis.trajectory_analysis import TrajectoryAnalyzer

print(f"CUDA Disponibile: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"Dispositivo CUDA: {torch.cuda.get_device_name(0)}")
    print(f"Numero di GPU: {torch.cuda.device_count()}")

class BarbellTracker:
    def __init__(self, video_path, output_path, confidence=0.5, class_id=None, device='cuda', 
                 output_dir=None):
        self.video_path = video_path
        self.output_path = output_path
        self.confidence = confidence
        self.device = device
        
        # Create output directory for analysis results
        if output_dir is None:
            self.output_dir = os.path.splitext(output_path)[0] + "_analysis"
        else:
            self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

        if self.device == 'cuda' and not torch.cuda.is_available():
            print("AVVISO: CUDA non è disponibile. Usando CPU.")
            self.device = 'cpu'
        
        print(f"Usando dispositivo: {self.device}")
        
        # Inizializzazione del modello YOLOv8 con il dispositivo specificato
        self.model = YOLO('yolov8n.pt').to(self.device)
        
        # Inizializzazione del tracker DeepSORT
        if self.device == 'cuda':
            self.tracker = DeepSort(
                max_age=30,
                nn_budget=100,
                embedder="mobilenet",  # Modello più leggero, adatto per GPU
                embedder_gpu=True      # Usa GPU per l'embedding
            )
        else:
            self.tracker = DeepSort(max_age=30)
        
        # Classe del bilanciere
        self.barbell_class_id = class_id
        
        # Dizionario per memorizzare le traiettorie
        self.trajectories = defaultdict(list)
        
        # Inizializzazione dell'analizzatore di traiettorie
        _, _, fps = self.getVideoProperties()
        self.analyzer = TrajectoryAnalyzer(fps=fps)
        
    def getVideoProperties(self):
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            print(f"Errore: Impossibile aprire il video {self.video_path}")
            return None, None, None
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        return width, height, fps

    def process_video(self):
        # Apertura del video
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            print(f"Errore: Impossibile aprire il video {self.video_path}")
            return
        
        # Ottenere le proprietà del video
        width, height, fps = self.getVideoProperties()
        if width is None:
            return
        
        # Configurazione dell'output video
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(self.output_path, fourcc, fps, (width, height))
        
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            print(f"Processing frame {frame_count}")
            
            # Usa il dispositivo specificato
            if self.barbell_class_id is not None:
                results = self.model(frame, classes=[self.barbell_class_id], device=self.device)
            else:
                results = self.model(frame, device=self.device)
            
            # Estrazione delle bounding box per i bilancieri
            detections = []
            
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    # Verifica se l'oggetto è un bilanciere (o simile) e se la confidenza è sufficiente
                    cls = int(box.cls[0])
                    conf = float(box.conf[0])
                    
                    if conf > self.confidence:
                        # Per ora, consideramo tutte le classi come potenziali bilancieri
                        # In un'implementazione reale, dovresti filtrare solo per bilancieri
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        
                        # Aggiungi rilevamento per DeepSORT
                        detections.append(([x1, y1, x2-x1, y2-y1], conf, cls))
            
            # Aggiornamento del tracker DeepSORT
            tracks = self.tracker.update_tracks(detections, frame=frame)
            
            # Aggiornamento delle traiettorie e disegno su frame
            for track in tracks:
                if not track.is_confirmed():
                    continue
                
                track_id = track.track_id
                ltrb = track.to_ltrb()
                
                x1, y1, x2, y2 = int(ltrb[0]), int(ltrb[1]), int(ltrb[2]), int(ltrb[3])
                
                # Calcolo del punto centrale dell'oggetto
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2
                
                # Aggiunta del punto alla traiettoria
                self.trajectories[track_id].append((center_x, center_y))
                
                # Disegno della bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"ID: {track_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
                # Disegno della traiettoria
                points = self.trajectories[track_id]
                for i in range(1, len(points)):
                    # Colore che cambia gradualmente lungo la traiettoria
                    color_intensity = min(255, i * 5)
                    cv2.line(frame, points[i-1], points[i], (0, 0, color_intensity), 2)
            
            # Scrittura del frame nel video di output
            out.write(frame)
            
            # Visualizzazione 
            cv2.imshow('Barbell Tracking', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        # Pulizia
        cap.release()
        out.release()
        cv2.destroyAllWindows()
        print(f"Video elaborato salvato come {self.output_path}")
        
        # Analyze trajectories after processing
        self.analyze_trajectories()
        
    def analyze_trajectories(self):
        """Analyze all tracked trajectories and generate visualizations."""
        print("\nAnalyzing trajectories...")
        
        # Find the longest trajectory
        longest_track_id = None
        max_length = 0
        
        for track_id, trajectory in self.trajectories.items():
            if len(trajectory) > max_length:
                max_length = len(trajectory)
                longest_track_id = track_id
                
        if longest_track_id is None:
            print("No trajectories to analyze!")
            return
            
        print(f"Analyzing trajectory of object with ID: {longest_track_id} (length: {max_length} frames)")
        
        # Analyze the longest trajectory
        trajectory = self.trajectories[longest_track_id]
        analysis_results = self.analyzer.analyze_trajectory(trajectory)
        
        if analysis_results is None:
            print("Failed to analyze trajectory.")
            return
            
        # Generate timestamp for output files
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create trajectory plot
        trajectory_plot_path = os.path.join(self.output_dir, f"trajectory_2d_{timestamp}.png")
        self.analyzer.plot_trajectory_2d(
            analysis_results["positions"], 
            title="Barbell Path", 
            save_path=trajectory_plot_path
        )
        print(f"Trajectory plot saved to {trajectory_plot_path}")
        
        # Create metrics plots
        metrics_plot_dir = self.output_dir
        self.analyzer.plot_metrics(
            analysis_results,
            title_prefix="Barbell",
            save_dir=metrics_plot_dir
        )
        print(f"Metrics plots saved to {metrics_plot_dir}")
        
        # Create animation
        animation_path = os.path.join(self.output_dir, f"trajectory_animation_{timestamp}.gif")
        self.analyzer.create_animation(
            trajectory,
            analysis_results,
            title="Barbell Motion Analysis",
            save_path=animation_path,
            fps=10
        )
        print(f"Animation saved to {animation_path}")
        
        # Export data to CSV
        csv_dir = os.path.join(self.output_dir, "csv_data")
        self.analyzer.export_to_csv(analysis_results, csv_dir)
        
        # Print summary statistics
        print("\nSummary Statistics:")
        print(f"Max Velocity: {analysis_results['max_velocity']:.2f} pixels/sec")
        print(f"Max Acceleration: {analysis_results['max_acceleration']:.2f} pixels/sec²")
        print(f"Average Velocity: {analysis_results['avg_velocity']:.2f} pixels/sec")
        print(f"Total Displacement: {analysis_results['total_displacement']:.2f} pixels")
        print(f"Total Path Length: {analysis_results['total_path_length']:.2f} pixels")
        
        # Save summary to text file
        summary_path = os.path.join(self.output_dir, f"summary_{timestamp}.txt")
        with open(summary_path, 'w') as f:
            f.write("Barbell Trajectory Analysis Summary\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write(f"Video: {os.path.basename(self.video_path)}\n")
            f.write(f"Object ID: {longest_track_id}\n")
            f.write(f"Trajectory Length: {max_length} frames\n\n")
            f.write("Statistics:\n")
            f.write(f"- Max Velocity: {analysis_results['max_velocity']:.2f} pixels/sec\n")
            f.write(f"- Max Acceleration: {analysis_results['max_acceleration']:.2f} pixels/sec²\n")
            f.write(f"- Average Velocity: {analysis_results['avg_velocity']:.2f} pixels/sec\n")
            f.write(f"- Total Displacement: {analysis_results['total_displacement']:.2f} pixels\n")
            f.write(f"- Total Path Length: {analysis_results['total_path_length']:.2f} pixels\n")
        
        print(f"\nAnalysis complete! Results saved to {self.output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Traccia la traiettoria di un bilanciere in un video.')
    parser.add_argument('--input', required=True, help='Percorso del video di input')
    parser.add_argument('--output', required=True, help='Percorso del video di output')
    parser.add_argument('--confidence', type=float, default=0.5, help='Soglia di confidenza per il rilevamento (default: 0.5)')
    parser.add_argument('--class_id', type=int, help='ID della classe da rilevare (se non specificato, rileva tutte le classi)')
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'], help='Dispositivo da utilizzare (default: cuda)')
    parser.add_argument('--output_dir', type=str, help='Directory per i risultati dell\'analisi (default: basato sul nome del video di output)')
    
    args = parser.parse_args()
    
    tracker = BarbellTracker(args.input, args.output, args.confidence, args.class_id, args.device, args.output_dir)
    tracker.process_video()
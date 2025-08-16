from datetime import datetime
import os
import cv2
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
import tqdm

class Visualizer:
    def plot_trajectory_on_video(trajectory, video_path, output_path=None, 
                             point_color=(0, 0, 255), point_size=5, 
                             draw_path=True, path_color=(255, 0, 0),
                             path_thickness=2, path_history=None,
                             show_velocity=False, analysis_results=None):
        """
        Disegna la traiettoria del bilanciere su ogni frame del video.
        
        Args:
            trajectory: Lista di punti (x,y) della traiettoria
            video_path: Percorso del video su cui disegnare la traiettoria
            output_path: Percorso dove salvare il video risultante
            point_color: Colore del punto (BGR) (default: rosso)
            point_size: Dimensione del punto (default: 5)
            draw_path: Se True, disegna anche il percorso completo
            path_color: Colore del percorso (BGR) (default: blu)
            path_thickness: Spessore del percorso (default: 2)
            path_history: Numero di punti storici da mostrare (None = tutti)
            show_velocity: Se True, mostra la velocità se disponibile
            analysis_results: Dizionario con i risultati dell'analisi
            
        Returns:
            Percorso del video risultante
        """
    
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Il file video {video_path} non esiste")
        
        # Apri il video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Impossibile aprire il video {video_path}")
        
        # Ottieni proprietà del video
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Crea il percorso di output se non specificato
        if output_path is None:
            base_name, ext = os.path.splitext(video_path)
            output_path = f"{base_name}_tracked{ext}"
        
        # Configura il writer per il video di output
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        # Converti la traiettoria in array NumPy
        # trajectory_array = np.array(trajectory, dtype=np.int32)
        
        # Ottieni velocità se disponibile
        velocities = None
        if show_velocity and analysis_results and 'velocities' in analysis_results:
            velocities = analysis_results['velocities']
        
        # Processa il video frame per frame
        pbar = tqdm(total=min(frame_count, len(trajectory)), desc="Elaborazione video")
        frame_idx = 0
        
        while True:
            ret, frame = cap.read()
            if not ret or frame_idx >= len(trajectory):
                break
                
            # Ottieni la posizione corrente
            x, y = trajectory[frame_idx]
            
            # Disegna il percorso
            if draw_path:
                # Determina quanti punti mostrare
                start_idx = 0 if path_history is None else max(0, frame_idx - path_history)
                
                # Disegna solo la parte di percorso necessaria
                for i in range(start_idx + 1, frame_idx + 1):
                    p1 = tuple(trajectory[i-1])
                    p2 = tuple(trajectory[i])
                    cv2.line(frame, p1, p2, path_color, path_thickness)
            
            # Disegna il punto della posizione attuale
            cv2.circle(frame, (x, y), point_size, point_color, -1)
            
            # Informazioni sul frame
            text = f"Frame: {frame_idx+1}/{len(trajectory)}"
            cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                        0.7, (255, 255, 255), 2)
            
            # Mostra la velocità se disponibile
            if velocities is not None and frame_idx < len(velocities):
                vel_text = f"Velocità: {velocities[frame_idx]:.2f} px/s"
                cv2.putText(frame, vel_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 
                            0.7, (255, 255, 255), 2)
            
            # Scrivi il frame nel video di output
            out.write(frame)
            
            # Aggiorna indice frame e progress bar
            frame_idx += 1
            pbar.update(1)
        
        # Rilascia le risorse
        pbar.close()
        cap.release()
        out.release()
        
        print(f"Video con traiettoria salvato in: {output_path}")
        return output_path

    def plot_trajectory_2d(trajectory, title="Barbell Trajectory", save_path=None):
            """
            Create 2D plot of the barbell path.
            
            Args:
                trajectory: List of (x, y) position tuples
                title: Title for the plot
                save_path: Path to save the plot (if None, plot is displayed)
            """
            trajectory_array = np.array(trajectory)
            
            # Create figure
            plt.figure(figsize=(10, 8))
            plt.plot(trajectory_array[:, 0], trajectory_array[:, 1], 'b-')
            plt.plot(trajectory_array[0, 0], trajectory_array[0, 1], 'go', label='Start')
            plt.plot(trajectory_array[-1, 0], trajectory_array[-1, 1], 'ro', label='End')
            
            # Invert y-axis to match image coordinates (y increases downward)
            plt.gca().invert_yaxis()
            
            plt.title(title)
            plt.xlabel('X Position (pixels)')
            plt.ylabel('Y Position (pixels)')
            plt.legend()
            plt.grid(True)
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                plt.close()
            else:
                plt.show()


    def plot_metrics(analysis_results, title_prefix="Barbell", save_dir=None):
            """
            Create plots for position, velocity and acceleration.
            
            Args:
                analysis_results: Dictionary from analyze_trajectory method
                title_prefix: Prefix for plot titles
                save_dir: Directory to save plots (if None, plots are displayed)
            """
            # Position over time
            fig, axs = plt.subplots(3, 1, figsize=(12, 18), sharex=True)
            
            # Position plot
            axs[0].plot(analysis_results["time"], analysis_results["positions"][:, 1], 'b-', label='Y Position')
            axs[0].set_title(f"{title_prefix} Vertical Position Over Time")
            axs[0].set_ylabel('Y Position (pixels)')
            axs[0].grid(True)
            axs[0].legend()
            # Invert y axis to match image coordinates
            axs[0].invert_yaxis()
            
            # Velocity plot
            axs[1].plot(analysis_results["velocity_time"], analysis_results["velocity_mag"], 'g-', label='Speed')
            axs[1].set_title(f"{title_prefix} Speed Over Time")
            axs[1].set_ylabel('Speed (pixels/sec)')
            axs[1].grid(True)
            axs[1].legend()
            
            # Acceleration plot
            axs[2].plot(analysis_results["acceleration_time"], analysis_results["acceleration_mag"], 'r-', label='Acceleration')
            axs[2].axhline(y=0, color='k', linestyle='--', alpha=0.3)
            axs[2].set_title(f"{title_prefix} Acceleration Over Time")
            axs[2].set_xlabel('Time (seconds)')
            axs[2].set_ylabel('Acceleration (pixels/sec²)')
            axs[2].grid(True)
            axs[2].legend()
            
            plt.tight_layout()
            
            if save_dir:
                os.makedirs(save_dir, exist_ok=True)
                filename = f"{title_prefix.lower().replace(' ', '_')}_metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
                plt.savefig(os.path.join(save_dir, filename), dpi=300, bbox_inches='tight')
                plt.close()
            else:
                plt.show()


    def create_animation(trajectory, analysis_results, title="Barbell Trajectory", save_path=None, fps=10):
            """
            Crea un'animazione della traiettoria del bilanciere.
            
            Args:
                trajectory: Array NumPy o lista di tuple (x, y) che rappresenta la traiettoria
                analysis_results: Dizionario con i risultati dell'analisi
                title: Titolo dell'animazione
                save_path: Percorso dove salvare l'animazione (GIF)
                fps: Frame per secondo dell'animazione
            """
            
            # Converti sempre in array NumPy
            trajectory_array = np.array(trajectory)
            
            fig, ax = plt.subplots(figsize=(10, 8))
            ax.set_title(title)
            
            # Disegna il percorso completo
            ax.plot(trajectory_array[:, 0], trajectory_array[:, 1], 'b-', alpha=0.3)
            
            # Inizializza il punto che si muoverà lungo la traiettoria
            point, = ax.plot([], [], 'ro', markersize=10)
            
            # Testo per velocità e accelerazione
            velocity_text = ax.text(0.02, 0.95, '', transform=ax.transAxes)
            
            # Imposta i limiti del grafico
            margin = 50  # margine in pixel
            ax.set_xlim(trajectory_array[:, 0].min() - margin, trajectory_array[:, 0].max() + margin)
            ax.set_ylim(trajectory_array[:, 1].min() - margin, trajectory_array[:, 1].max() + margin)
            
            # Inverti l'asse y per corrispondere alle coordinate immagine
            ax.invert_yaxis()
            
            # Funzione di aggiornamento chiamata per ogni frame
            def update(frame):
                # Assicurati che frame sia un indice valido
                frame = min(frame, len(trajectory_array) - 1)
                
                # CORREZIONE QUI: Passa liste, non singoli valori
                point.set_data([trajectory_array[frame, 0]], [trajectory_array[frame, 1]])
                
                # Aggiorna testo con dati di velocità se disponibili
                if 'velocities' in analysis_results and len(analysis_results['velocities']) > frame:
                    vel = analysis_results['velocities'][frame]
                    velocity_text.set_text(f'Velocità: {vel:.2f} px/s')
                
                return point, velocity_text
            
            # Crea l'animazione
            frames = len(trajectory_array)
            ani = FuncAnimation(fig, update, frames=range(frames), interval=1000/fps, blit=True)
            
            # Salva o mostra l'animazione
            if save_path:
                # Assicurati che la directory esista
                import os
                os.makedirs(os.path.dirname(os.path.abspath(save_path)) if os.path.dirname(save_path) else '.', exist_ok=True)
                
                ani.save(save_path, writer='pillow', fps=fps)
                print(f"Animazione salvata in: {save_path}")
            else:
                plt.show()


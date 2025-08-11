import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from matplotlib.animation import FuncAnimation
import pandas as pd
import os
from datetime import datetime

class TrajectoryAnalyzer:
    def __init__(self, fps=30):
        self.fps = fps
        self.time_delta = 1.0 / fps
        
    def smooth_trajectory(self, trajectory, window_size=15, poly_order=3):
        """
        Apply Savitzky-Golay filter to smooth the trajectory data.
        
        Args:
            trajectory: List of (x, y) position tuples
            window_size: Window size for the filter (must be odd and >= poly_order+2)
            poly_order: Polynomial order for the filter
            
        Returns:
            Smoothed trajectory as numpy array with shape (n, 2)
        """
        if len(trajectory) < window_size:
            # If trajectory is too short, adjust window size
            window_size = max(5, len(trajectory) - 2)
            # Make sure window size is odd
            window_size = window_size if window_size % 2 == 1 else window_size - 1
            # Make sure window size is at least poly_order + 2
            if window_size <= poly_order + 1:
                window_size = poly_order + 2
                if window_size % 2 == 0:
                    window_size += 1
        
        # Extract x and y coordinates
        x_coords = np.array([pos[0] for pos in trajectory])
        y_coords = np.array([pos[1] for pos in trajectory])
        
        # Apply Savitzky-Golay filter for smoothing
        try:
            x_smooth = savgol_filter(x_coords, window_size, poly_order)
            y_smooth = savgol_filter(y_coords, window_size, poly_order)
            return np.column_stack((x_smooth, y_smooth))
        except Exception as e:
            print(f"Smoothing failed: {e}. Returning original trajectory.")
            return np.array(trajectory)
    
    def calculate_velocity(self, smooth_trajectory):
        """
        Calculate velocity from smoothed trajectory.
        
        Args:
            smooth_trajectory: Numpy array of smoothed (x, y) positions
            
        Returns:
            Velocities as numpy array with shape (n-1, 2)
        """
        # Calculate velocity using central differences
        velocity = np.diff(smooth_trajectory, axis=0) / self.time_delta
        return velocity
    
    def calculate_acceleration(self, velocity):
        """
        Calculate acceleration from velocity.
        
        Args:
            velocity: Numpy array of (vx, vy) velocities
            
        Returns:
            Accelerations as numpy array with shape (n-2, 2)
        """
        # Calculate acceleration using central differences
        acceleration = np.diff(velocity, axis=0) / self.time_delta
        return acceleration
    
    def calculate_magnitude(self, vectors):
        """
        Calculate magnitude of vectors.
        
        Args:
            vectors: Numpy array of vectors (vx, vy) or (ax, ay)
            
        Returns:
            Magnitudes as numpy array
        """
        return np.sqrt(np.sum(vectors**2, axis=1))
    
    def analyze_trajectory(self, trajectory):
        """
        Perform full analysis on trajectory.
        
        Args:
            trajectory: List of (x, y) position tuples
            
        Returns:
            Dictionary containing analysis results
        """
        if len(trajectory) < 5:
            print("Trajectory too short for analysis.")
            return None
        
        # Convert trajectory to numpy array
        trajectory_array = np.array(trajectory)
        
        # Create time array
        time_array = np.arange(len(trajectory_array)) * self.time_delta
        
        # Smooth trajectory
        smooth_trajectory = self.smooth_trajectory(trajectory_array)
        
        # Calculate velocity and acceleration
        velocity = self.calculate_velocity(smooth_trajectory)
        acceleration = self.calculate_acceleration(velocity)
        
        # Calculate magnitudes
        velocity_mag = self.calculate_magnitude(velocity)
        acceleration_mag = self.calculate_magnitude(acceleration)
        
        # Time arrays for velocity and acceleration
        velocity_time = time_array[:-1]
        acceleration_time = time_array[:-2]
        
        # Calculate statistics
        max_velocity = np.max(velocity_mag)
        max_acceleration = np.max(acceleration_mag)
        avg_velocity = np.mean(velocity_mag)
        
        # Displacement (distance from starting to ending point)
        total_displacement = np.linalg.norm(
            trajectory_array[-1] - trajectory_array[0]
        )
        
        # Path length (sum of all segments)
        segments = trajectory_array[1:] - trajectory_array[:-1]
        segment_lengths = np.sqrt(np.sum(segments**2, axis=1))
        total_path_length = np.sum(segment_lengths)
        
        return {
            "time": time_array,
            "positions": smooth_trajectory,
            "velocity_time": velocity_time,
            "velocity": velocity,
            "velocity_mag": velocity_mag,
            "acceleration_time": acceleration_time,
            "acceleration": acceleration,
            "acceleration_mag": acceleration_mag,
            "max_velocity": max_velocity,
            "max_acceleration": max_acceleration,
            "avg_velocity": avg_velocity,
            "total_displacement": total_displacement,
            "total_path_length": total_path_length
        }
        
    def plot_trajectory_2d(self, trajectory, title="Barbell Trajectory", save_path=None):
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
    
    def plot_metrics(self, analysis_results, title_prefix="Barbell", save_dir=None):
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
    
    def create_animation(self, trajectory, analysis_results, title="Barbell Trajectory", save_path=None, fps=10):
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
            
    def export_to_csv(self, analysis_results, save_path):
        """
        Export analysis results to CSV files.
        
        Args:
            analysis_results: Dictionary from analyze_trajectory method
            save_path: Directory to save CSV files
        """
        os.makedirs(save_path, exist_ok=True)
        
        # Export position data
        position_df = pd.DataFrame({
            'time': analysis_results["time"],
            'x_position': analysis_results["positions"][:, 0],
            'y_position': analysis_results["positions"][:, 1]
        })
        position_df.to_csv(os.path.join(save_path, "position_data.csv"), index=False)
        
        # Export velocity data
        velocity_df = pd.DataFrame({
            'time': analysis_results["velocity_time"],
            'x_velocity': analysis_results["velocity"][:, 0],
            'y_velocity': analysis_results["velocity"][:, 1],
            'velocity_magnitude': analysis_results["velocity_mag"]
        })
        velocity_df.to_csv(os.path.join(save_path, "velocity_data.csv"), index=False)
        
        # Export acceleration data
        acceleration_df = pd.DataFrame({
            'time': analysis_results["acceleration_time"],
            'x_acceleration': analysis_results["acceleration"][:, 0],
            'y_acceleration': analysis_results["acceleration"][:, 1],
            'acceleration_magnitude': analysis_results["acceleration_mag"]
        })
        acceleration_df.to_csv(os.path.join(save_path, "acceleration_data.csv"), index=False)
        
        # Export summary statistics
        stats_df = pd.DataFrame({
            'metric': ['max_velocity', 'max_acceleration', 'avg_velocity', 
                      'total_displacement', 'total_path_length'],
            'value': [analysis_results["max_velocity"], 
                     analysis_results["max_acceleration"],
                     analysis_results["avg_velocity"],
                     analysis_results["total_displacement"],
                     analysis_results["total_path_length"]]
        })
        stats_df.to_csv(os.path.join(save_path, "summary_stats.csv"), index=False)
        
        print(f"Data exported to {save_path}")
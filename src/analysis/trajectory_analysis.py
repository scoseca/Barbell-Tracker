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
    
    def create_animation(self, trajectory, analysis_results, title="Barbell Motion", save_path=None, fps=10):
        """
        Create an animated visualization of barbell motion with metrics.
        
        Args:
            trajectory: Original list of (x, y) position tuples
            analysis_results: Dictionary from analyze_trajectory method
            title: Title for the animation
            save_path: Path to save the animation (if None, animation is displayed)
            fps: Frames per second for the animation
        """
        fig = plt.figure(figsize=(15, 10))
        
        # Create grid layout
        gs = fig.add_gridspec(2, 2)
        ax1 = fig.add_subplot(gs[0, 0])  # Trajectory
        ax2 = fig.add_subplot(gs[0, 1])  # Y position over time
        ax3 = fig.add_subplot(gs[1, 0])  # Velocity over time
        ax4 = fig.add_subplot(gs[1, 1])  # Acceleration over time
        
        # Set up trajectory plot
        trajectory_array = np.array(trajectory)
        ax1.plot(trajectory_array[:, 0], trajectory_array[:, 1], 'b-', alpha=0.5)
        point, = ax1.plot([], [], 'ro', markersize=10)
        ax1.set_title('Barbell Path')
        ax1.set_xlabel('X Position (pixels)')
        ax1.set_ylabel('Y Position (pixels)')
        ax1.invert_yaxis()  # Invert y-axis to match image coordinates
        ax1.grid(True)
        
        # Set up position plot
        pos_line, = ax2.plot([], [], 'b-')
        pos_point, = ax2.plot([], [], 'bo', markersize=8)
        ax2.set_title('Vertical Position')
        ax2.set_xlabel('Time (seconds)')
        ax2.set_ylabel('Y Position (pixels)')
        ax2.grid(True)
        ax2.invert_yaxis()  # Invert y-axis to match image coordinates
        
        # Set up velocity plot
        vel_line, = ax3.plot([], [], 'g-')
        vel_point, = ax3.plot([], [], 'go', markersize=8)
        ax3.set_title('Speed')
        ax3.set_xlabel('Time (seconds)')
        ax3.set_ylabel('Speed (pixels/sec)')
        ax3.grid(True)
        
        # Set up acceleration plot
        acc_line, = ax4.plot([], [], 'r-')
        acc_point, = ax4.plot([], [], 'ro', markersize=8)
        ax4.set_title('Acceleration')
        ax4.set_xlabel('Time (seconds)')
        ax4.set_ylabel('Acceleration (pixels/sec²)')
        ax4.axhline(y=0, color='k', linestyle='--', alpha=0.3)
        ax4.grid(True)
        
        # Set axis limits
        ax2.set_xlim(0, analysis_results["time"][-1])
        ax2.set_ylim(np.min(analysis_results["positions"][:, 1]) * 1.1, 
                    np.max(analysis_results["positions"][:, 1]) * 0.9)  # Inverted y-axis
        
        ax3.set_xlim(0, analysis_results["velocity_time"][-1])
        ax3.set_ylim(0, np.max(analysis_results["velocity_mag"]) * 1.1)
        
        ax4.set_xlim(0, analysis_results["acceleration_time"][-1])
        max_acc = np.max(np.abs(analysis_results["acceleration_mag"]))
        ax4.set_ylim(-max_acc * 1.1, max_acc * 1.1)
        
        # Add title to the figure
        fig.suptitle(title, fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust layout to make room for the figure title
        
        def update(frame):
            # Update trajectory point
            if frame < len(trajectory_array):
                point.set_data(trajectory_array[frame, 0], trajectory_array[frame, 1])
            
            # Update position plot
            pos_line.set_data(analysis_results["time"][:frame+1], 
                             analysis_results["positions"][:frame+1, 1])
            if frame < len(analysis_results["positions"]):
                pos_point.set_data(analysis_results["time"][frame], 
                                  analysis_results["positions"][frame, 1])
            
            # Update velocity plot
            if frame < len(analysis_results["velocity_time"]):
                vel_line.set_data(analysis_results["velocity_time"][:frame+1], 
                                analysis_results["velocity_mag"][:frame+1])
                vel_point.set_data(analysis_results["velocity_time"][frame], 
                                  analysis_results["velocity_mag"][frame])
            
            # Update acceleration plot
            if frame < len(analysis_results["acceleration_time"]):
                acc_line.set_data(analysis_results["acceleration_time"][:frame+1], 
                                analysis_results["acceleration_mag"][:frame+1])
                acc_point.set_data(analysis_results["acceleration_time"][frame], 
                                 analysis_results["acceleration_mag"][frame])
            
            return point, pos_line, pos_point, vel_line, vel_point, acc_line, acc_point
        
        # Create animation
        frames = min(len(trajectory_array), len(analysis_results["time"]))
        ani = FuncAnimation(fig, update, frames=frames, interval=1000/fps, blit=True)
        
        if save_path:
            ani.save(save_path, writer='pillow', fps=fps)
            plt.close()
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
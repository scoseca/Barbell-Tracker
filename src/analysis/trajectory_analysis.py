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
                    
    
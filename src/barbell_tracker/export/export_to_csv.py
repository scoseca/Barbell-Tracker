import os
import pandas as pd


def export_to_csv(analysis_results, save_path):
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
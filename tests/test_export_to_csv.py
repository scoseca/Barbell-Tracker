import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.barbell_tracker.export.export_to_csv import export_to_csv


class ExportToCsvTest(unittest.TestCase):
    def test_export_to_csv_writes_expected_files_and_columns(self):
        analysis_results = {
            "time": np.array([0.0, 0.5, 1.0]),
            "positions": np.array([[10.0, 20.0], [11.0, 22.0], [12.0, 24.0]]),
            "velocity_time": np.array([0.0, 0.5]),
            "velocity": np.array([[2.0, 4.0], [2.0, 4.0]]),
            "velocity_mag": np.array([4.4721, 4.4721]),
            "acceleration_time": np.array([0.0]),
            "acceleration": np.array([[0.0, 0.0]]),
            "acceleration_mag": np.array([0.0]),
            "max_velocity": 4.4721,
            "max_acceleration": 0.0,
            "avg_velocity": 4.4721,
            "total_displacement": 4.4721,
            "total_path_length": 4.4721,
        }

        with tempfile.TemporaryDirectory() as tmp_dir:
            export_to_csv(analysis_results, tmp_dir)
            output_dir = Path(tmp_dir)

            self.assertEqual(
                {
                    path.name
                    for path in output_dir.iterdir()
                },
                {
                    "position_data.csv",
                    "velocity_data.csv",
                    "acceleration_data.csv",
                    "summary_stats.csv",
                },
            )

            with (output_dir / "position_data.csv").open(newline="") as csv_file:
                rows = list(csv.DictReader(csv_file))

            self.assertEqual(
                rows[0],
                {"time": "0.0", "x_position": "10.0", "y_position": "20.0"},
            )

            with (output_dir / "summary_stats.csv").open(newline="") as csv_file:
                summary_rows = list(csv.DictReader(csv_file))

            self.assertEqual(summary_rows[0]["metric"], "max_velocity")
            self.assertEqual(summary_rows[-1]["metric"], "total_path_length")


if __name__ == "__main__":
    unittest.main()

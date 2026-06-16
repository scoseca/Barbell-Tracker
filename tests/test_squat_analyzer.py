import unittest

import numpy as np

from src.barbell_tracker.analysis.squat_analyzer import SquatAnalyzer


class SquatAnalyzerTest(unittest.TestCase):
    def test_analyze_squat_calculates_deviation_metrics(self):
        analyzer = SquatAnalyzer(safe_zone_px=20)
        trajectory = np.array(
            [
                [90.0, 10.0],
                [100.0, 20.0],
                [115.0, 30.0],
                [80.0, 40.0],
            ]
        )

        results = analyzer.analyze_squat(trajectory, foot_position=(100.0, 0.0))

        self.assertAlmostEqual(results["mean_deviation_px"], -3.75)
        self.assertAlmostEqual(results["max_forward_deviation_px"], 15.0)
        self.assertAlmostEqual(results["max_backward_deviation_px"], -20.0)
        self.assertAlmostEqual(results["std_deviation_px"], np.std([-10, 0, 15, -20]))
        self.assertAlmostEqual(results["percent_in_safe_zone"], 75.0)
        self.assertEqual(results["max_forward_point"], (115.0, 30.0))
        self.assertEqual(results["max_backward_point"], (80.0, 40.0))
        self.assertTrue(results["assessment"].startswith("Accettabile"))


if __name__ == "__main__":
    unittest.main()

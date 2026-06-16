import unittest

import numpy as np

from src.barbell_tracker.analysis.trajectory_analysis import TrajectoryAnalyzer


class TrajectoryAnalyzerTest(unittest.TestCase):
    def test_analyze_linear_trajectory_returns_expected_metrics(self):
        analyzer = TrajectoryAnalyzer(fps=2)
        trajectory = [(float(i), float(2 * i)) for i in range(8)]

        results = analyzer.analyze_trajectory(trajectory)

        self.assertIsNotNone(results)
        self.assertEqual(results["positions"].shape, (8, 2))
        self.assertEqual(results["velocity"].shape, (7, 2))
        self.assertEqual(results["acceleration"].shape, (6, 2))
        np.testing.assert_allclose(results["time"], np.arange(8) * 0.5)
        np.testing.assert_allclose(
            results["velocity_mag"],
            np.full(7, 2 * np.sqrt(5)),
            atol=1e-10,
        )
        self.assertAlmostEqual(results["max_acceleration"], 0.0, places=10)
        self.assertAlmostEqual(results["total_path_length"], 7 * np.sqrt(5))
        self.assertAlmostEqual(results["total_displacement"], np.sqrt(7**2 + 14**2))

    def test_short_trajectory_returns_none(self):
        analyzer = TrajectoryAnalyzer()

        self.assertIsNone(analyzer.analyze_trajectory([(0, 0), (1, 1)]))


if __name__ == "__main__":
    unittest.main()

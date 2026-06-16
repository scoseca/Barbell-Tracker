import sys
import tempfile
import types
import unittest
from pathlib import Path

import cv2
import numpy as np


class FakeImageFormat:
    SRGB = "SRGB"


class FakeImage:
    def __init__(self, image_format, data):
        self.image_format = image_format
        self.data = data


fake_mediapipe = types.SimpleNamespace(Image=FakeImage, ImageFormat=FakeImageFormat)
sys.modules.setdefault("mediapipe", fake_mediapipe)


class FakeKMeans:
    def __init__(self, n_clusters, random_state=None, n_init=None):
        self.n_clusters = n_clusters

    def fit(self, points):
        if self.n_clusters != 1:
            raise ValueError("FakeKMeans supports only one cluster in these tests")
        self.labels_ = np.zeros(len(points), dtype=int)
        self.cluster_centers_ = np.array([np.mean(points, axis=0)])
        return self


fake_sklearn = types.ModuleType("sklearn")
fake_sklearn_cluster = types.ModuleType("sklearn.cluster")
fake_sklearn_cluster.KMeans = FakeKMeans
fake_sklearn.cluster = fake_sklearn_cluster
sys.modules.setdefault("sklearn", fake_sklearn)
sys.modules.setdefault("sklearn.cluster", fake_sklearn_cluster)

from src.barbell_tracker.pose.foot_detector import FootPositionDetector


class Landmark:
    def __init__(self, x, y):
        self.x = x
        self.y = y


class FakePoseLandmarker:
    def __init__(self):
        self.landmarks = [Landmark(0.0, 0.0) for _ in range(33)]
        self.landmarks[29] = Landmark(0.10, 0.25)
        self.landmarks[31] = Landmark(0.20, 0.25)
        self.landmarks[30] = Landmark(0.60, 0.70)
        self.landmarks[32] = Landmark(0.70, 0.70)

    def detect(self, image):
        return types.SimpleNamespace(pose_landmarks=[self.landmarks])


class FootPositionDetectorTest(unittest.TestCase):
    def test_extract_foot_positions_uses_stable_synthetic_landmarks(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            video_path = Path(tmp_dir) / "synthetic.avi"
            self._write_video(video_path, width=80, height=80, frame_count=12)

            detector = FootPositionDetector()
            foot_position = detector.extract_foot_positions_from_video(
                str(video_path),
                FakePoseLandmarker(),
                n_clusters=1,
                visualize=False,
            )

        self.assertEqual(tuple(foot_position), (12.0, 20.0))

    def test_extract_foot_positions_falls_back_to_frame_center_without_pose(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            video_path = Path(tmp_dir) / "synthetic.avi"
            self._write_video(video_path, width=64, height=48, frame_count=3)

            detector = FootPositionDetector()
            foot_position = detector.extract_foot_positions_from_video(
                str(video_path),
                types.SimpleNamespace(detect=lambda image: types.SimpleNamespace(pose_landmarks=[])),
                visualize=False,
            )

        self.assertEqual(foot_position, (32, 24))

    @staticmethod
    def _write_video(video_path, width, height, frame_count):
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        writer = cv2.VideoWriter(str(video_path), fourcc, 10, (width, height))
        if not writer.isOpened():
            raise RuntimeError(f"Unable to create test video at {video_path}")

        frame = np.zeros((height, width, 3), dtype=np.uint8)
        for _ in range(frame_count):
            writer.write(frame)
        writer.release()


if __name__ == "__main__":
    unittest.main()

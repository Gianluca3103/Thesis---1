"""LiDAR reflectivity must reach the shared sparse encoder in every frame."""

import unittest

import numpy as np
import torch

from models.radar_lidar_stage2.sparse_batch import voxelize_lidar_features
from scripts.predict_stage2_sparse import _present_in_faulty


class TestStage2LidarFeatures(unittest.TestCase):
    def test_reflectivity_mean_and_batch_separation(self):
        points = torch.tensor([
            [1.01, 0.01, 0.01, 0.0],
            [1.09, 0.09, 0.09, 255.0],
            [1.01, 0.01, 0.01, 255.0],
        ])
        batches = torch.tensor([0, 0, 1])
        coords, features = voxelize_lidar_features(points, batches)
        self.assertEqual(coords.tolist(), [[0, 8, 80, 5], [1, 8, 80, 5]])
        torch.testing.assert_close(features[0, :3],
                                   torch.tensor([1.05, 0.05, 0.05]))
        torch.testing.assert_close(features[:, 3], torch.tensor([0.5, 1.0]))

    def test_empty_fault_cloud(self):
        coords, features = voxelize_lidar_features(torch.empty((0, 4)))
        self.assertEqual(tuple(coords.shape), (0, 4))
        self.assertEqual(tuple(features.shape), (0, 4))

    def test_fog_reflectivity_change_is_visible(self):
        clean = np.array([[1.0, 2.0, 3.0, 120.0]], dtype=np.float32)
        fog = np.array([[1.0, 2.0, 3.0, 28.0]], dtype=np.float32)
        self.assertTrue(_present_in_faulty(clean[:, :3], fog[:, :3]).item())
        self.assertFalse(_present_in_faulty(clean, fog).item())
        self.assertFalse(_present_in_faulty(clean, fog[:0]).item())


if __name__ == "__main__":
    unittest.main()

"""LiDAR reflectivity must reach the shared sparse encoder in every frame."""

import unittest

import numpy as np
import torch

from models.radar_lidar_stage2.sparse_batch import voxelize_lidar_features
from models.radar_lidar_stage2.fault_data import decode_fog_points, encode_fog_points
from models.radar_lidar_stage2.point_distribution import points_on_allowed
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
        self.assertEqual(tuple(features.shape), (2, 8))
        torch.testing.assert_close(features[:, 7], torch.log1p(torch.tensor([2.0, 1.0])))
        self.assertGreater(float(features[0, 4:7].sum()), 0)

    def test_empty_fault_cloud(self):
        coords, features = voxelize_lidar_features(torch.empty((0, 4)))
        self.assertEqual(tuple(coords.shape), (0, 4))
        self.assertEqual(tuple(features.shape), (0, 8))

    def test_multiple_clean_points_keep_one_allowed_site(self):
        clean = torch.tensor([[1.01, 0.01, 0.01],
                              [1.09, 0.09, 0.09],
                              [1.41, 0.01, 0.01]])
        allowed = torch.tensor([[0, 8, 80, 5]])
        sites, local, counts = points_on_allowed(clean, allowed)
        self.assertEqual(sites.tolist(), [0, 0])
        self.assertEqual(counts.tolist(), [2])
        self.assertEqual(tuple(local.shape), (2, 3))
        self.assertTrue(bool((local.abs() <= 1).all()))

    def test_same_voxel_in_two_frames_keeps_point_counts_separate(self):
        clean = torch.tensor([[1.01, 0.01, 0.01],
                              [1.09, 0.09, 0.09],
                              [1.04, 0.04, 0.04]])
        allowed = torch.tensor([[0, 8, 80, 5], [1, 8, 80, 5]])
        batches = torch.tensor([0, 0, 1])
        sites, local, counts = points_on_allowed(clean, allowed, batches)
        self.assertEqual(sites.tolist(), [0, 0, 1])
        self.assertEqual(counts.tolist(), [2, 1])
        self.assertEqual(tuple(local.shape), (3, 3))

    def test_fog_reflectivity_change_is_visible(self):
        clean = np.array([[1.0, 2.0, 3.0, 120.0]], dtype=np.float32)
        fog = np.array([[1.0, 2.0, 3.0, 28.0]], dtype=np.float32)
        self.assertTrue(_present_in_faulty(clean[:, :3], fog[:, :3]).item())
        self.assertFalse(_present_in_faulty(clean, fog).item())
        self.assertFalse(_present_in_faulty(clean, fog[:0]).item())

    def test_fog_cache_shuffle_preserves_float32_bits(self):
        points = np.array([[1.125, -0.0, 3.5, 28.0],
                           [-7.25, 8.0, -0.125, 120.75]], dtype=np.float32)
        restored = decode_fog_points(encode_fog_points(points), len(points))
        np.testing.assert_array_equal(restored.view(np.uint32),
                                      points.view(np.uint32))


if __name__ == "__main__":
    unittest.main()

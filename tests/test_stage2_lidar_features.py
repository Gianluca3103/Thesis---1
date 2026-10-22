"""Stage II encodes XYZ only and fog retains only original LiDAR returns."""

import unittest

import numpy as np
import torch

from models.radar_lidar_stage2.fault_data import (
    decode_fog_points, encode_fog_points, retain_original_fog_returns,
    severe_fault,
)
from models.radar_lidar_stage2.sparse_batch import voxelize_means_batched


class TestStage2LidarFeatures(unittest.TestCase):
    def test_xyz_means_and_batch_separation(self):
        points = torch.tensor([[1.01, 0.01, 0.01],
                               [1.09, 0.09, 0.09],
                               [1.01, 0.01, 0.01]])
        coords, features = voxelize_means_batched(
            points, torch.tensor([0, 0, 1]))
        self.assertEqual(coords.tolist(), [[0, 8, 80, 5], [1, 8, 80, 5]])
        torch.testing.assert_close(features[0], torch.tensor([1.05, 0.05, 0.05]))
        self.assertEqual(tuple(features.shape), (2, 3))
        with self.assertRaisesRegex(ValueError, "XYZ only"):
            voxelize_means_batched(torch.cat((points, points[:, :1]), 1),
                                   torch.tensor([0, 0, 1]))

    def test_empty_fault_cloud(self):
        coords, features = voxelize_means_batched(
            torch.empty((0, 3)), torch.empty(0, dtype=torch.long))
        self.assertEqual(tuple(coords.shape), (0, 4))
        self.assertEqual(tuple(features.shape), (0, 3))

    def test_fog_drops_shifted_returns_and_extra_duplicates(self):
        clean = np.array([[1, 2, 3], [1, 2, 3], [4, 5, 6]], dtype=np.float32)
        cached = np.array([[1, 2, 3, 10], [9, 9, 9, 100],
                           [1, 2, 3, 20], [1, 2, 3, 30],
                           [4, 5, 6, 40]], dtype=np.float32)
        retained = retain_original_fog_returns(clean, cached)
        np.testing.assert_array_equal(retained, clean)
        self.assertEqual(retained.shape[1], 3)

    def test_fog_fault_exposes_only_original_xyz(self):
        clean = torch.tensor([[1.0, 2.0, 3.0, 120.0],
                              [4.0, 5.0, 6.0, 80.0]])
        class Cache:
            def load(self, _frame_id):
                return torch.tensor([[1.0, 2.0, 3.0],
                                     [8.0, 9.0, 10.0]])
        faulty, name = severe_fault(clean, "00001", 0, Cache(),
                                    fault_type="fog_sim")
        self.assertEqual(name, "fog_sim")
        torch.testing.assert_close(faulty, clean[:1, :3])

    def test_fog_cache_shuffle_preserves_float32_bits(self):
        points = np.array([[1.125, -0.0, 3.5, 28.0],
                           [-7.25, 8.0, -0.125, 120.75]], dtype=np.float32)
        restored = decode_fog_points(encode_fog_points(points), len(points))
        np.testing.assert_array_equal(restored.view(np.uint32),
                                      points.view(np.uint32))


if __name__ == "__main__":
    unittest.main()

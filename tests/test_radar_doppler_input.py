"""Radar Doppler pooling and radar-only PFE input compatibility."""

import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from torch import nn

from models.radar_lidar_crossmodal_v1.encoder import OriginalPFE, radar_voxelize_batch


class _FakeSuppliedPFE(nn.Module):
    def __init__(self, num_input_features=4, latent_dim=64):
        super().__init__()
        self.input_mlp = nn.Sequential(nn.Linear(4, 16), nn.ReLU())


class RadarDopplerInputTests(unittest.TestCase):
    def test_radial_and_compensated_velocities_reach_voxel_features(self):
        radar = torch.tensor([
            [0.01, 0., -1.95, 100., 2., -1., 0.],
            [0.09, 0., -1.95, -100., 6., 3., -19.],
            [0.21, 0., -1.95, 7., -4., 8., -3.],
            [90., 0., 0., 7., 99., 99., 0.],
        ], requires_grad=True)
        features, coords = radar_voxelize_batch([radar])
        self.assertEqual(features.shape, (2, 6))
        self.assertEqual(coords.shape, (2, 4))
        self.assertEqual(coords[:, 0].tolist(), [0, 0])
        torch.testing.assert_close(features[:, 4:], torch.tensor([[4., 1.], [-4., 8.]]))
        torch.testing.assert_close(features[:, 3], torch.tensor([2., 1.]).log())
        features[:, 4:].sum().backward()
        self.assertTrue(torch.isfinite(radar.grad).all())
        self.assertNotEqual(radar.grad[:3, 4:6].abs().sum().item(), 0.)
        self.assertEqual(radar.grad[:, [3, 6]].abs().sum().item(), 0.)

    def test_only_radar_changes_first_pfe_layer_and_checkpoints_cannot_cross_load(self):
        fake_vendor = types.SimpleNamespace(PFE_Engine=_FakeSuppliedPFE)
        with patch("torch.cuda.is_available", return_value=True), patch(
                "models.radar_lidar_crossmodal_v1.encoder._load_vendor", return_value=fake_vendor):
            radar = OriginalPFE(modality="radar")
            lidar = OriginalPFE(modality="lidar")
            self.assertEqual(radar.encoder.input_mlp[0].in_features, 6)
            self.assertEqual(lidar.encoder.input_mlp[0].in_features, 4)
            with tempfile.TemporaryDirectory() as directory:
                radar_path = Path(directory) / "radar.pth"
                lidar_path = Path(directory) / "lidar.pth"
                torch.save(radar.encoder.state_dict(), radar_path)
                torch.save(lidar.encoder.state_dict(), lidar_path)
                OriginalPFE(radar_path, modality="radar")
                OriginalPFE(lidar_path, modality="lidar")
                with self.assertRaisesRegex(ValueError, "radar checkpoint must have 6"):
                    OriginalPFE(lidar_path, modality="radar")
                with self.assertRaisesRegex(ValueError, "lidar checkpoint must have 4"):
                    OriginalPFE(radar_path, modality="lidar")


if __name__ == "__main__":
    unittest.main()

"""Contracts for the fixed Stage-II gate and the ray/offset-only objective."""

import unittest
from unittest.mock import patch

import numpy as np
import torch

from models.radar_lidar_crossmodal_v1.radar_pruner import SHAPE
from models.radar_lidar_stage2.geometry_gate import geometry_gate_output, offset_targets
from models.radar_lidar_stage2.sparse_geometry import GRID, select_allowed
from models.radar_lidar_stage2.sparse_residual import (
    SparseResidualStage2, Stage2Conditions,
)
from scripts.evaluate_stage2_geometry_baseline import _counts, _select


class GeometryGateTest(unittest.TestCase):
    def test_only_supported_unobserved_cells_are_selected(self):
        mask = np.zeros(SHAPE, dtype=bool)
        mask[8, 80, 80:83] = True
        radar = torch.tensor([[16.1, 0.1, 0.1, 4, 2, 3, 1]],
                             dtype=torch.float32)
        # Occupy the middle supported voxel; all other faulty cells must not
        # be added and unsupported cells must never enter the gate.
        faulty = torch.tensor([[16.3, 0.1, 0.1], [16.9, 0.1, 0.1]],
                              dtype=torch.float32)
        output = geometry_gate_output(radar, faulty, mask)
        fine, scores = select_allowed(output["confidence_coords"],
                                      output["confidence_probabilities"], 0.5)
        self.assertEqual(fine[:, 1:].tolist(), [[8, 80, 80], [8, 80, 82]])
        self.assertTrue(torch.equal(scores, torch.ones_like(scores)))
        self.assertEqual(output["conditioning_features"].shape[1], 64)

    def test_empty_radar_has_no_learned_fallback(self):
        output = geometry_gate_output(torch.empty((0, 7)), torch.empty((0, 3)),
                                      np.zeros(SHAPE, dtype=bool))
        self.assertEqual(len(output["confidence_coords"]), 0)
        self.assertEqual(len(output["conditioning_coords"]), 0)

    def test_offset_target_is_centroid_relative_to_eligible_cell(self):
        fine = torch.tensor([[0, 8, 80, 80], [0, 8, 80, 81]])
        clean = torch.tensor([[16.02, 0.02, 0.02], [16.08, 0.08, 0.08]])
        occupied, offsets, coverage = offset_targets(clean, fine)
        self.assertEqual(occupied.tolist(), [True, False])
        self.assertTrue(torch.allclose(offsets[0], torch.tensor([-0.05, -0.05, -0.075]),
                                       atol=1e-5))
        self.assertEqual(coverage, 1.0)


class ObjectiveTest(unittest.TestCase):
    def test_calibration_threshold_is_applied_to_heldout_counts(self):
        calibration_pos = np.array([1, 2, 7, 10], dtype=np.int64)
        calibration_neg = np.array([20, 10, 1, 0], dtype=np.int64)
        choices = _select(calibration_pos, calibration_neg, 20)
        threshold_bin = choices["best_f1"]
        heldout = _counts(np.array([1, 1, 8, 10]),
                          np.array([10, 9, 2, 0]), 20, 25,
                          threshold_bin)
        self.assertEqual(heldout["eligible_target"], 20)
        self.assertEqual(heldout["full_missing_target"], 25)
        self.assertAlmostEqual(heldout["full_missing_recall"],
                               heldout["tp"] / 25)

    def test_ray_offset_mode_never_calls_addition_or_healthy_loss(self):
        class Fake:
            diffusion_target = "epsilon"
            grid = GRID

            def __init__(self):
                self.weight = torch.nn.Parameter(torch.tensor(0.2))

            def encode_lidar(self, xyz, conditions, batch_ids):
                latent = xyz.new_zeros((1, 32))
                return latent, torch.zeros(1, dtype=torch.bool)

            def diffusion(self, noisy, faulty, radar, probability, present,
                          timestep, coords):
                return noisy * 0 + self.weight

            def lpcr(self, clean_hat, hierarchy, probabilities,
                     return_hidden=False):
                logits = clean_hat[:, 0]
                offsets = clean_hat[:, :3]
                hidden = clean_hat[:, :16]
                return logits, offsets, hidden

            def lpcr_loss(self, *args, **kwargs):
                raise AssertionError("Ray/offset mode called LPCR occupancy loss")

            def addition_head(self, hidden, logits, probability, observed, density):
                return logits

            def addition_loss(self, *args, **kwargs):
                raise AssertionError("Ray/offset mode called addition loss")

        fake = Fake()
        fine = torch.tensor([[0, 8, 80, 80]], dtype=torch.long)
        conditions = Stage2Conditions((fine, fine, fine, fine, fine),
                                      torch.ones((1, 1)), torch.ones((1, 1)),
                                      torch.zeros((1, 32)))
        ray = lambda fine, logits, observed, occupied, clean, **kwargs: {
            "hit": logits.square().mean(), "extra": logits.square().mean(),
            "hit_rays": 1, "extra_sites": 1}
        with patch("models.radar_lidar_stage2.sparse_residual.addition_ray_losses", ray):
            result = SparseResidualStage2.diffusion_loss(
                fake, torch.empty((0, 3)),
                torch.tensor([[16.12, 0.13, 0.12]]), conditions,
                torch.linspace(0.99, 0.01, 10), ray_offset_baseline=True,
                ray_hit_weight=1, ray_extra_weight=1, offset_weight=1)
        self.assertGreater(float(result["reconstruction"]["offset"].detach()), 0)
        expected = (result["noise"] + result["task_weight"] * (
            result["reconstruction"]["offset"] + result["rays"]["hit"]
            + result["rays"]["extra"]))
        self.assertTrue(torch.allclose(result["loss"], expected))
        result["loss"].backward()
        self.assertIsNotNone(fake.weight.grad)
        self.assertTrue(torch.isfinite(fake.weight.grad))


if __name__ == "__main__":
    unittest.main()

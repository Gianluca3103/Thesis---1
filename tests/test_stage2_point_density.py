"""The variable-cardinality point head stays inside Stage-I-allowed voxels."""

import math
import unittest
from types import SimpleNamespace

import torch
from torch import nn

from models.radar_lidar_stage2.sparse_geometry import GRID
from models.radar_lidar_stage2.sparse_residual import (
    AllowedSiteLPCR, POINT_MIXTURES, SparseResidualStage2, Stage2Conditions,
)


class TestStage2PointDensity(unittest.TestCase):
    def _head(self) -> AllowedSiteLPCR:
        # Test the point head on CPU without instantiating sparse convolutions.
        head = AllowedSiteLPCR.__new__(AllowedSiteLPCR)
        nn.Module.__init__(head)
        head.grid = GRID
        head.point_density = nn.Linear(16, POINT_MIXTURES * 7)
        return head

    def test_emits_three_points_in_one_allowed_voxel(self):
        torch.manual_seed(3103)
        head = self._head()
        hidden = torch.randn((2, 16))
        allowed = torch.tensor([[0, 8, 80, 5], [0, 8, 80, 6]])
        logits = torch.tensor([2.0, -2.0])
        log_count = torch.tensor([math.log1p(3), math.log1p(7)])
        output = head.emit_points(hidden, allowed, logits, log_count, 0.5)
        self.assertEqual(output["allowed_cells"], 2)
        self.assertEqual(output["predicted_cells"], 1)
        self.assertEqual(output["predicted_points"], 3)
        self.assertEqual(output["point_counts"].tolist(), [3])
        self.assertEqual(output["coordinates"].tolist(), [allowed[0].tolist()] * 3)
        xyz = output["points_xyz"]
        self.assertTrue(bool((xyz[:, 0] >= 1.0).all() and (xyz[:, 0] < 1.2).all()))
        self.assertTrue(bool((xyz[:, 1] >= 0).all() and (xyz[:, 1] < 0.2).all()))
        self.assertTrue(bool((xyz[:, 2] >= 0).all() and (xyz[:, 2] < 0.25).all()))

    def test_point_likelihood_has_gradient(self):
        torch.manual_seed(3103)
        head = self._head()
        hidden = torch.randn((1, 16), requires_grad=True)
        site_index = torch.tensor([0, 0, 0])
        local = torch.tensor([[-0.8, 0.0, 0.2], [0.0, 0.3, -0.4],
                              [0.7, -0.5, 0.1]])
        loss = head.point_nll(hidden, site_index, local, torch.tensor([3]),
                              torch.tensor([0]), 1)
        loss.backward()
        self.assertTrue(bool(torch.isfinite(loss)))
        self.assertIsNotNone(head.point_density.weight.grad)
        self.assertTrue(bool(torch.isfinite(head.point_density.weight.grad).all()))
        self.assertTrue(bool((head.point_density.weight.grad != 0).any()))

    def test_distribution_centroid_has_gradient(self):
        torch.manual_seed(3103)
        head = self._head()
        hidden = torch.randn((2, 16), requires_grad=True)
        predicted = head.expected_local_centroid(hidden)
        self.assertEqual(tuple(predicted.shape), (2, 3))
        self.assertTrue(bool((predicted.abs() <= 1).all()))
        loss = (predicted - torch.tensor([[0.5, 0.0, -0.5],
                                          [-0.5, 0.0, 0.5]])).abs().mean()
        loss.backward()
        self.assertTrue(bool(torch.isfinite(head.point_density.weight.grad).all()))
        self.assertTrue(bool((head.point_density.weight.grad != 0).any()))

    def test_lpcr_loss_uses_all_points_in_occupied_voxel(self):
        head = self._head()
        allowed = torch.tensor([[0, 8, 80, 5], [0, 8, 80, 6]])
        logits = torch.tensor([2.0, -2.0], requires_grad=True)
        log_count = torch.tensor([math.log1p(3), 0.0], requires_grad=True)
        hidden = torch.randn((2, 16), requires_grad=True)
        head.forward = lambda *_args: (logits, log_count, hidden)
        conditions = Stage2Conditions(
            (allowed,) * 5, torch.ones((2, 1)), torch.ones((2, 1)),
            torch.empty((2, 32)),
            (torch.tensor([True, False]), torch.tensor([False, True]),
             torch.zeros((2, 3)), 1.0))
        clean = torch.tensor([[1.01, 0.01, 0.01],
                              [1.04, 0.04, 0.04],
                              [1.08, 0.08, 0.08]])
        stub = SimpleNamespace(grid=GRID, lpcr=head)
        result = SparseResidualStage2.lpcr_loss(
            stub, torch.empty((2, 32)), conditions, clean)
        result["loss"].backward()
        self.assertEqual(result["target_points"], 3.0)
        self.assertEqual(result["positive_sites"], 1.0)
        self.assertLess(result["count_mae"], 0.01)
        torch.testing.assert_close(
            result["loss"], result["occupancy"] + 0.1 * result["count"]
            + 0.04 * result["point_nll"] + 0.04 * result["centroid"])
        self.assertTrue(bool(torch.isfinite(head.point_density.weight.grad).all()))


if __name__ == "__main__":
    unittest.main()

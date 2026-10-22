"""The restored LPCR head emits one offset point per occupied voxel."""

import unittest
from types import SimpleNamespace

import torch
from torch import nn
from torch.nn import functional as F

from models.radar_lidar_stage2.sparse_geometry import GRID
from models.radar_lidar_stage2.sparse_residual import (
    AllowedSiteLPCR, SparseResidualStage2, Stage2Conditions,
)


class TestStage2Offset(unittest.TestCase):
    def _head(self):
        head = AllowedSiteLPCR.__new__(AllowedSiteLPCR)
        nn.Module.__init__(head)
        head.grid = GRID
        return head

    def test_one_point_per_selected_voxel(self):
        head = self._head()
        coords = torch.tensor([[0, 8, 80, 5], [0, 8, 80, 6]])
        output = head.emit_points(coords, torch.tensor([2.0, -2.0]),
                                  torch.zeros((2, 3)), 0.5)
        self.assertEqual(output["allowed_cells"], 2)
        self.assertEqual(output["predicted_cells"], 1)
        self.assertEqual(output["predicted_points"], 1)
        self.assertEqual(output["point_counts"].tolist(), [1])
        self.assertEqual(output["coordinates"].tolist(), [coords[0].tolist()])
        torch.testing.assert_close(output["points_xyz"][0],
                                   torch.tensor([1.1, 0.1, 0.125]))

    def test_old_occupancy_plus_offset_loss(self):
        head = self._head()
        logits = torch.tensor([2.0, -2.0], requires_grad=True)
        offsets = torch.tensor([[0.02, -0.01, 0.03], [0.0, 0.0, 0.0]],
                               requires_grad=True)
        head.forward = lambda *_args: (logits, offsets)
        allowed = torch.tensor([[0, 8, 80, 5], [0, 8, 80, 6]])
        conditions = Stage2Conditions(
            (allowed,) * 5, torch.ones((2, 1)), torch.ones((2, 1)),
            torch.empty((2, 32)),
            (torch.tensor([True, False]), torch.tensor([False, True]),
             torch.zeros((2, 3)), 1.0))
        stub = SimpleNamespace(grid=GRID, lpcr=head)
        result = SparseResidualStage2.lpcr_loss(
            stub, torch.empty((2, 32)), conditions, torch.empty((0, 3)))
        expected = F.binary_cross_entropy_with_logits(
            logits, torch.tensor([1.0, 0.0])) + 0.04 * offsets[0].abs().mean()
        torch.testing.assert_close(result["loss"], expected)
        result["loss"].backward()
        self.assertIsNotNone(logits.grad)
        self.assertIsNotNone(offsets.grad)


if __name__ == "__main__":
    unittest.main()

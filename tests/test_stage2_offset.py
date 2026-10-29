"""R2LDM-style Stage-II occupancy and centroid-offset objective."""

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

    def test_one_centroid_per_selected_allowed_voxel(self):
        head = self._head()
        coords = torch.tensor([[0, 8, 80, 5], [0, 8, 80, 6]])
        offsets = torch.tensor([[0.02, -0.01, 0.03], [0., 0., 0.]])
        output = head.emit_points(coords, torch.tensor([2., -2.]),
                                  offsets, 0.5)
        self.assertEqual(output["allowed_cells"], 2)
        self.assertEqual(output["predicted_cells"], 1)
        self.assertEqual(output["predicted_points"], 1)
        self.assertEqual(output["point_counts"].tolist(), [1])
        self.assertEqual(output["coordinates"].tolist(), [coords[0].tolist()])
        torch.testing.assert_close(output["points_xyz"][0],
                                   torch.tensor([1.12, 0.09, 0.155]))

    def test_duplicate_cell_keeps_highest_confidence_and_offset_stays_inside(self):
        head = self._head()
        coords = torch.tensor([[0, 8, 80, 5], [0, 8, 80, 5],
                               [0, 8, 80, 6]])
        logits = torch.tensor([0.2, 2.0, 1.0])
        offsets = torch.tensor([[0.0, 0.0, 0.0],
                                [0.10, 0.10, 0.125],
                                [-0.10, -0.10, -0.125]])
        output = head.emit_points(coords, logits, offsets, 0.5)
        self.assertEqual(output["predicted_points"], 2)
        self.assertEqual(output["point_counts"].tolist(), [1, 1])
        self.assertEqual(output["coordinates"].tolist(),
                         [coords[0].tolist(), coords[2].tolist()])
        torch.testing.assert_close(output["probabilities"][0],
                                   logits[1].sigmoid())
        lower = torch.tensor(GRID.minimum_xyz)
        size = torch.tensor(GRID.voxel_xyz)
        actual_xyz = torch.floor((output["points_xyz"] - lower) / size).long()
        torch.testing.assert_close(actual_xyz[:, [2, 1, 0]],
                                   output["coordinates"][:, 1:])

    def test_paper_weighted_occupancy_and_centroid_offset_loss(self):
        head = self._head()
        logits = torch.tensor([2., -2.], requires_grad=True)
        offsets = torch.zeros((2, 3), requires_grad=True)
        head.forward = lambda *_args: (logits, offsets)
        allowed = torch.tensor([[0, 8, 80, 5], [0, 8, 80, 6]])
        clean = torch.tensor([[1.12, 0.09, 0.03]])
        target_offsets = torch.tensor([[0.02, 0.03, -0.01], [0., 0., 0.]])
        conditions = Stage2Conditions(
            (allowed,) * 5, torch.ones((2, 1)), torch.ones((2, 1)),
            torch.empty((2, 32)),
            (torch.tensor([True, False]), torch.tensor([False, True]),
             target_offsets, 1.0))
        stub = SimpleNamespace(grid=GRID, lpcr=head)
        result = SparseResidualStage2.lpcr_loss(
            stub, torch.empty((2, 32)), conditions, clean)
        expected = (0.9 * F.binary_cross_entropy_with_logits(
            logits, torch.tensor([1., 0.]))
            + 0.1 * target_offsets[0].abs().mean())
        torch.testing.assert_close(result["loss"], expected)
        result["loss"].backward()
        for tensor in (logits, offsets):
            self.assertIsNotNone(tensor.grad)
            self.assertTrue(bool((tensor.grad != 0).any()))


if __name__ == "__main__":
    unittest.main()

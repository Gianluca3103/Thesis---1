"""The Stage-II LPCR decoder learns one to five XYZ points per occupied voxel."""

import unittest
from types import SimpleNamespace

import torch
from torch import nn
from torch.nn import functional as F

from models.radar_lidar_stage2.point_targets import multipoint_targets
from models.radar_lidar_stage2.sparse_geometry import GRID
from models.radar_lidar_stage2.sparse_residual import (
    AllowedSiteLPCR, SparseResidualStage2, Stage2Conditions,
    _matched_point_offset_loss,
)


class TestStage2Offset(unittest.TestCase):
    def _head(self):
        head = AllowedSiteLPCR.__new__(AllowedSiteLPCR)
        nn.Module.__init__(head)
        head.grid = GRID
        return head

    def test_predicted_count_is_bounded_and_repeats_only_allowed_coordinates(self):
        head = self._head()
        coords = torch.tensor([[0, 8, 80, 5], [0, 8, 80, 6]])
        counts = torch.tensor([[0., 0., 3., 0., 0.],
                               [0., 0., 0., 0., 4.]])
        offsets = torch.zeros((2, 5, 3))
        output = head.emit_points(coords, torch.tensor([2., -2.]),
                                  counts, offsets, 0.5)
        self.assertEqual(output["allowed_cells"], 2)
        self.assertEqual(output["predicted_cells"], 1)
        self.assertEqual(output["predicted_points"], 3)
        self.assertEqual(output["point_counts"].tolist(), [3])
        self.assertEqual(output["coordinates"].tolist(), [coords[0].tolist()] * 3)
        torch.testing.assert_close(output["points_xyz"][0],
                                   torch.tensor([1.1, 0.1, 0.125]))
        five = head.emit_points(coords[:1], torch.tensor([2.]),
                                counts[1:], offsets[:1], 0.5)
        self.assertEqual(five["predicted_points"], 5)

    def test_targets_cap_at_five_and_keep_batches_separate(self):
        coords = torch.tensor([[0, 8, 80, 5], [1, 8, 80, 5]])
        clean = torch.tensor([[1.01 + 0.01 * index, 0.01, 0.01]
                              for index in range(7)] + [[1.03, 0.02, 0.02]])
        batch_ids = torch.tensor([0] * 7 + [1])
        counts, offsets = multipoint_targets(clean, coords, batch_ids)
        self.assertEqual(counts.tolist(), [5, 1])
        self.assertEqual(offsets.shape, (2, 5, 3))
        self.assertTrue(bool((offsets.abs() <= offsets.new_tensor(
            GRID.voxel_xyz) * 0.5).all()))

    def test_offset_matching_does_not_depend_on_slot_order(self):
        target = torch.zeros((1, 5, 3))
        target[0, 0, 0] = 0.04
        target[0, 1, 1] = -0.03
        predicted = target.clone()
        predicted[:, :2] = target[:, [1, 0]]
        loss = _matched_point_offset_loss(
            predicted, target, torch.tensor([2]), torch.tensor([0]), 1)
        torch.testing.assert_close(loss, torch.tensor(0.))

    def test_occupancy_count_and_weighted_offset_loss(self):
        head = self._head()
        logits = torch.tensor([2., -2.], requires_grad=True)
        count_logits = torch.zeros((2, 5), requires_grad=True)
        offsets = torch.zeros((2, 5, 3), requires_grad=True)
        head.forward = lambda *_args: (logits, count_logits, offsets)
        allowed = torch.tensor([[0, 8, 80, 5], [0, 8, 80, 6]])
        clean = torch.tensor([[1.12, 0.09, 0.03], [1.08, 0.11, 0.06]])
        conditions = Stage2Conditions(
            (allowed,) * 5, torch.ones((2, 1)), torch.ones((2, 1)),
            torch.empty((2, 32)),
            (torch.tensor([True, False]), torch.tensor([False, True]),
             torch.zeros((2, 3)), 1.0))
        stub = SimpleNamespace(grid=GRID, lpcr=head)
        result = SparseResidualStage2.lpcr_loss(
            stub, torch.empty((2, 32)), conditions, clean)
        _, targets = multipoint_targets(clean, allowed)
        expected_offset = targets[0, :2].abs().mean()
        expected = (F.binary_cross_entropy_with_logits(
            logits, torch.tensor([1., 0.]))
            + 0.1 * F.cross_entropy(count_logits[:1], torch.tensor([1]))
            + 0.1 * expected_offset)
        torch.testing.assert_close(result["loss"], expected)
        result["loss"].backward()
        for tensor in (logits, count_logits, offsets):
            self.assertIsNotNone(tensor.grad)
            self.assertTrue(bool((tensor.grad != 0).any()))


if __name__ == "__main__":
    unittest.main()

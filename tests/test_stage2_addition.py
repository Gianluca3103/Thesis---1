"""Addition decisions protect observed regions without using clean input."""

import unittest

import torch
from torch import nn

from models.radar_lidar_stage2.addition import (
    addition_context, healthy_region_negatives,
)
from models.radar_lidar_stage2.sparse_geometry import GRID
from models.radar_lidar_stage2.sparse_residual import (
    AdditionHead, AllowedSiteLPCR, SparseResidualStage2, Stage2Conditions,
)


def xyz(coords):
    index = coords[:, [3, 2, 1]].float()
    return (index + 0.5) * torch.tensor(GRID.voxel_xyz) + torch.tensor(
        GRID.minimum_xyz)


class TestStage2Addition(unittest.TestCase):
    def test_healthy_region_has_negative_empty_cells_but_keeps_missing_positive(self):
        clean_coords = torch.tensor(
            [[0, 0, y, x] for x0 in (0, 4) for y in (0, 1)
             for x in range(x0, x0 + 4)], dtype=torch.long)
        # Seven of eight clean cells survive in the first 0.8 m region.
        faulty_coords = clean_coords[:7]
        empty_healthy = torch.tensor([[0, 0, 3, 3]])
        empty_lost = torch.tensor([[0, 0, 3, 7]])
        allowed = torch.cat((clean_coords, empty_healthy, empty_lost))
        clean_occupied = torch.tensor([True] * len(clean_coords) + [False, False])
        context = addition_context(xyz(faulty_coords), allowed)
        healthy = healthy_region_negatives(
            xyz(clean_coords), allowed, clean_occupied, context)
        self.assertTrue(bool(healthy[-2]))
        self.assertFalse(bool(healthy[-1]))
        self.assertFalse(bool(healthy[:len(clean_coords)].any()))
        self.assertFalse(bool(context.observed[7]))
        self.assertGreater(float(context.local_faulty_density[-2]), 0)
        self.assertEqual(float(context.local_faulty_density[-1]), 0)

    def test_addition_loss_updates_head_and_full_cloud_features(self):
        clean_coords = torch.tensor(
            [[0, 0, y, x] for y in (0, 1) for x in range(4)], dtype=torch.long)
        faulty_coords = clean_coords[:7]
        empty = torch.tensor([[0, 0, 3, 3]])
        allowed = torch.cat((clean_coords, empty))
        clean_occupied = torch.tensor([True] * 8 + [False])
        conditions = Stage2Conditions(
            (allowed,) * 5, torch.full((9, 1), 0.8),
            torch.ones((9, 1)), torch.empty((9, 32)))
        stub = SparseResidualStage2.__new__(SparseResidualStage2)
        nn.Module.__init__(stub)
        stub.grid = GRID
        stub.addition_head = AdditionHead()
        hidden = torch.randn((9, 16), requires_grad=True)
        full_logits = torch.randn(9, requires_grad=True)
        result = stub.addition_loss(
            hidden, full_logits, xyz(faulty_coords), xyz(clean_coords),
            conditions, clean_occupied, torch.zeros(9, dtype=torch.bool))
        (result["final"] + result["healthy"]).backward()
        self.assertEqual(result["missing_sites"], 1.0)
        self.assertEqual(result["healthy_negative_sites"], 1.0)
        for grad in (hidden.grad, full_logits.grad,
                     stub.addition_head.network[0].weight.grad):
            self.assertIsNotNone(grad)
            self.assertTrue(bool(torch.isfinite(grad).all()))
            self.assertTrue(bool((grad != 0).any()))

    def test_sample_adds_only_unobserved_voxels_without_clean_input(self):
        class Decoder(nn.Module):
            emit_points = AllowedSiteLPCR.emit_points

            def __init__(self):
                super().__init__()
                self.grid = GRID

            def forward(self, latent, hierarchy, probability, *, return_hidden=False):
                count = len(hierarchy[0])
                logits = latent.new_full((count,), 4.0)
                offsets = latent.new_zeros((count, 3))
                hidden = latent.new_zeros((count, 16))
                return (logits, offsets, hidden) if return_hidden else (logits, offsets)

        class Head(nn.Module):
            def forward(self, hidden, full, radar, observed, density):
                return full

        allowed = torch.tensor([[0, 0, 0, 0], [0, 0, 0, 1]])
        conditions = Stage2Conditions(
            (allowed,) * 5, torch.full((2, 1), 0.8),
            torch.ones((2, 1)), torch.empty((2, 32)))
        stub = SparseResidualStage2.__new__(SparseResidualStage2)
        nn.Module.__init__(stub)
        stub.grid = GRID
        stub.lpcr = Decoder()
        stub.addition_head = Head()
        stub.use_addition_head = True
        stub.encode_lidar = lambda _xyz, _conditions: (
            torch.zeros((2, 32)), torch.zeros(2, dtype=torch.bool))
        output = stub.sample(xyz(allowed[:1]), conditions,
                             torch.tensor([0.9, 0.8]), steps=1,
                             use_residual=False)
        self.assertEqual(output["full_estimate_cells"], 2)
        self.assertEqual(output["overlap_observed_cells"], 1)
        self.assertEqual(output["coordinates"].tolist(), [allowed[1].tolist()])


if __name__ == "__main__":
    unittest.main()

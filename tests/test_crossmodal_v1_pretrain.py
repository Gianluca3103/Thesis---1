"""CPU checks for the added encoder pretraining objective."""

import torch

from models.radar_lidar_crossmodal_v1.encoder import GRID, masked_bottleneck_sites
from models.radar_lidar_crossmodal_v1.pretrain_loss import (
    observed_child_distribution, observed_distribution_loss,
)
from models.radar_lidar_stage1.sparse import SparseSites


def test_mask_is_shared_by_all_fine_voxels_of_a_bottleneck_site():
    coordinates = torch.tensor([[0, 0, 0, 0], [0, 7, 7, 7], [0, 0, 8, 0],
                                [1, 0, 0, 0]], dtype=torch.long)
    for salt in range(12):
        masked = masked_bottleneck_sites(coordinates, .3, salt)
        assert masked[0] == masked[1]
    assert not masked_bottleneck_sites(coordinates, 0., 0).any()


def test_observed_child_loss_uses_only_held_out_sites():
    sites = SparseSites(torch.tensor([[0, 0, 32, 32], [0, 0, 32, 33]]),
                        torch.zeros(2, 64), GRID.scale_shape(8), 8)
    points = torch.tensor([[[0.05, 0.05, -1.95], [1.65, 0.05, -1.95]]])
    counts = observed_child_distribution(sites, points,
                                         torch.ones((1, 2), dtype=torch.bool))
    assert counts.sum().item() == 2
    assert counts.sum(-1).tolist() == [1., 1.]
    logits = torch.zeros(2, 512, requires_grad=True)
    loss, eligible = observed_distribution_loss(logits, counts,
                                                torch.tensor([True, False]))
    assert eligible == 1
    torch.testing.assert_close(loss, torch.tensor(512.).log())
    loss.backward()
    assert logits.grad[0].abs().sum() > 0
    assert logits.grad[1].abs().sum() == 0

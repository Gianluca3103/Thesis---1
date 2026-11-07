"""The clean diagnostic must keep supervision and scoring inside the gate."""

import numpy as np
import torch

from models.radar_lidar_stage2.clean_denoising import (
    batched_hierarchy, clean_points_in_support, frame_mean,
    occupancy_counts, occupancy_metrics)
from models.radar_lidar_stage2.sparse_geometry import GRID


def test_clean_points_and_sparse_sites_stay_inside_support():
    mask = np.zeros(GRID.shape_zyx, dtype=np.bool_)
    mask[8, 80, 10] = True
    mask[8, 80, 11] = True
    xyz = torch.tensor([[2.1, 0.1, 0.1], [2.3, 0.1, 0.1],
                        [2.5, 0.1, 0.1], [-0.1, 0.1, 0.1]], dtype=torch.float32)
    filtered = clean_points_in_support(xyz, mask)
    assert torch.equal(filtered, xyz[:2])
    levels = batched_hierarchy(
        [torch.tensor([[8, 80, 10], [8, 80, 11]]),
         torch.tensor([[8, 80, 10]])], "cpu")
    assert levels[0].tolist() == [[0, 8, 80, 10], [0, 8, 80, 11],
                                    [1, 8, 80, 10]]
    assert set(levels[-1][:, 0].tolist()) == {0, 1}


def test_equal_frame_loss_and_eligible_occupancy_metrics():
    # One very large frame must not drown out another frame's loss.
    loss = frame_mean(torch.tensor([0.0, 0.0, 0.0, 2.0]),
                      torch.tensor([0, 0, 0, 1]), 2)
    assert loss.item() == 1.0
    counts = occupancy_counts(torch.tensor([10.0, -10.0, 10.0]),
                              torch.tensor([True, True, False]))
    metrics = occupancy_metrics(counts)
    assert counts == {"tp": 1, "predicted": 2, "target": 2}
    assert metrics["precision"] == 0.5
    assert metrics["recall"] == 0.5
    assert metrics["iou"] == 1 / 3

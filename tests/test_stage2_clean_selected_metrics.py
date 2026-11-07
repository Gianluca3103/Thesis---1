"""Selected clean occupancy metrics use centers and one-to-one tolerance."""

import pytest
import torch

from models.radar_lidar_stage2.clean_selected_metrics import (
    add_frame, empty_accumulator, summarize, voxel_center_chamfer)


def test_shifted_occupied_voxel_matches_within_20cm():
    target = torch.tensor([[0, 8, 80, 10]])
    predicted = torch.tensor([[0, 8, 80, 11]])
    counts = empty_accumulator()
    add_frame(counts, predicted, target)
    row = summarize(counts)
    assert row["iou"] == 0.0
    assert row["f1"] == 0.0
    assert row["iou_0p2m"] == 1.0
    assert row["f1_0p2m"] == 1.0
    assert row["chamfer_l2_m_paired_frame_mean"] == pytest.approx(0.4)
    assert row["chamfer_squared_l2_m2_paired_frame_mean"] == pytest.approx(0.08)


def test_empty_prediction_is_counted_and_chamfer_undefined():
    target = torch.tensor([[0, 8, 80, 10]])
    empty = torch.empty((0, 4), dtype=torch.long)
    assert voxel_center_chamfer(empty, target) is None
    counts = empty_accumulator()
    add_frame(counts, empty, target)
    row = summarize(counts)
    assert row["empty_prediction_frames"] == 1
    assert row["chamfer_paired_frames"] == 0
    assert row["chamfer_squared_l2_m2_paired_frame_mean"] is None
    assert row["f1_0p2m"] == 0.0

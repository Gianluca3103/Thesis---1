"""Selected clean occupancy metrics use centers and one-to-one tolerance."""

import pytest
import torch

from models.radar_lidar_stage2.clean_selected_metrics import (
    add_frame, completion_summary, empty_accumulator,
    full_frame_voxels, summarize, voxel_center_chamfer, voxel_centers)


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


def test_full_roi_completion_unions_faulty_and_gated_predictions():
    cells = torch.tensor([[0, 8, 80, 10], [0, 8, 80, 11]])
    clean_xyz = torch.from_numpy(voxel_centers(cells)).float()
    faulty_xyz = clean_xyz[:1]
    # The first selected cell already exists in faulty LiDAR. Count it once.
    clean, faulty, completed = full_frame_voxels(
        clean_xyz, faulty_xyz, cells[[0, 1, 1]])
    before, after = empty_accumulator(), empty_accumulator()
    add_frame(before, faulty, clean)
    add_frame(after, completed, clean)
    result = completion_summary(before, after)
    assert result["faulty"]["predicted"] == 1
    assert result["completed"]["predicted"] == 2
    assert result["faulty"]["iou"] == 0.5
    assert result["completed"]["iou"] == 1.0
    assert result["improvement"]["iou"] == 0.5
    assert result["improvement"]["f1_0p2m"] > 0


def test_full_roi_counts_false_additions_in_reliable_area():
    clean_cell = torch.tensor([[0, 8, 80, 10]])
    clean_xyz = torch.from_numpy(voxel_centers(clean_cell)).float()
    false_cell = torch.tensor([[0, 8, 80, 30]])
    clean, faulty, completed = full_frame_voxels(
        clean_xyz, clean_xyz, false_cell)
    before, after = empty_accumulator(), empty_accumulator()
    add_frame(before, faulty, clean)
    add_frame(after, completed, clean)
    result = completion_summary(before, after)
    assert result["faulty"]["iou"] == 1.0
    assert result["completed"]["iou"] == 0.5
    assert result["improvement"]["iou_0p2m"] == -0.5
    assert result["improvement"]["f1_0p2m"] < 0

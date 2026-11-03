"""Two-radius geometry baseline semantics and denominators."""

from __future__ import annotations

import json

import numpy as np
import torch

from models.radar_lidar_crossmodal_v1.radar_pruner import radar_support_mask
from scripts.compare_radar_geometry_radii import (
    _empty_counts, _selected_indices, _summarize, evaluate,
    third_nearest_distance)


def _radar(points):
    rows = torch.zeros((len(points), 7), dtype=torch.float32)
    if points:
        rows[:, :3] = torch.tensor(points)
    return rows


def test_distinct_voxels_and_nested_radii():
    assert np.isinf(third_nearest_distance(_radar([(10, 0, 0)] * 9))).all()
    radar = _radar([(10.0, 0, 0), (10.6, 0, 0), (11.2, 0, 0)])
    distances = third_nearest_distance(radar)
    near = distances <= 1.6 + 1e-9
    far = distances <= 3.2 + 1e-9
    assert near.any()
    np.testing.assert_array_equal(near, radar_support_mask(radar, 1.6))
    assert np.all(~near | far)
    old = radar_support_mask(radar)
    assert np.all(~near | old)
    assert np.all(~old | far)


def test_keep_all_precision_and_recall_denominators():
    count = _empty_counts()
    count.update(frames=2, radar_supported_cells=100,
                 clean_lidar_voxels=20, supported_clean_lidar_voxels=15,
                 missing_lidar_voxels=10, supported_missing_lidar_voxels=6,
                 addition_candidates=80)
    result = _summarize(count, has_faulty=True)
    assert result["clean_occupancy"]["precision"] == 0.15
    assert result["clean_occupancy"]["full_recall"] == 0.75
    assert result["missing_additions"]["precision"] == 0.075
    assert result["missing_additions"]["full_recall"] == 0.6
    assert result["missing_additions"]["eligible_recall"] == 1.0


def test_match_report_uses_heldout_ids(tmp_path):
    class Frame:
        def __init__(self, frame_id):
            self.frame_id = frame_id

    class Dataset:
        frames = [Frame("1"), Frame("2"), Frame("3"), Frame("4")]

    path = tmp_path / "matched.json"
    path.write_text(json.dumps({"sampled_frame_ids": ["3", "1", "4", "2"],
                                "calibration_frames": 2}), encoding="utf-8")
    assert _selected_indices(Dataset(), path, None) == [3, 1]


def test_faulty_lidar_addition_metric_uses_missing_targets(tmp_path):
    radar = _radar([(10.0, 0.0, 0.0), (10.6, 0.0, 0.0),
                    (11.2, 0.0, 0.0)])
    clean = torch.tensor([[10.5, 0.1, 0.125, 1.0],
                          [10.9, 0.1, 0.125, 1.0]])
    faulty_root = tmp_path / "faulty"
    (faulty_root / "val").mkdir(parents=True)
    clean[:1].numpy().tofile(faulty_root / "val" / "00001.bin")

    class Dataset:
        def __len__(self):
            return 1

        def __getitem__(self, index):
            return {"frame_id": "00001", "radar": radar,
                    "clean_lidar": clean}

    results = evaluate(Dataset(), [0], faulty_root,
                       {"00001": "partial_loss_40"})
    for radius in ("1.6", "3.2"):
        row = results["all"][radius]["missing_additions"]
        assert row["target"] == row["tp"] == 1
        assert row["eligible_recall"] == row["full_recall"] == 1.0
        assert row["selected"] > row["tp"]

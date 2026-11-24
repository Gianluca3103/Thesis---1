"""Conditioned samples show actual additions and their full-ROI errors."""

import json
import re

import numpy as np
import pytest
import torch

from models.radar_lidar_stage2.clean_selected_metrics import (
    add_frame, completion_summary, empty_accumulator, voxel_centers)
from scripts.visualize_stage2_conditioned_completion import (
    _save_record, _viewer, balanced_frame_ids)


def test_balanced_frame_ids_selects_each_fault_reproducibly():
    ids = [f"{index:05d}" for index in range(12)]
    faults = {frame_id: ("fog" if index % 2 else "partial")
              for index, frame_id in enumerate(ids)}
    chosen = balanced_frame_ids(ids, faults, 2, 3103)
    assert chosen == balanced_frame_ids(ids, faults, 2, 3103)
    assert len(chosen) == 4
    assert [faults[frame_id] for frame_id in chosen].count("fog") == 2
    assert [faults[frame_id] for frame_id in chosen].count("partial") == 2
    with pytest.raises(ValueError):
        balanced_frame_ids(ids, faults, 7, 3103)


def test_conditioned_sample_writes_completed_scan_and_light_viewer(tmp_path):
    clean = torch.tensor([[0, 8, 80, 10], [0, 8, 80, 11]])
    faulty = clean[:1]
    added = clean[1:]
    completed = clean
    before, after = empty_accumulator(), empty_accumulator()
    add_frame(before, faulty, clean)
    add_frame(after, completed, clean)
    record = {
        "frame_id": "00001", "split": "test", "fault": "partial_loss_30",
        "epoch": 76, "threshold": 0.81, "clean": clean,
        "faulty": faulty, "completed": completed, "added": added,
        "gate": clean, "probability": np.array([0.2, 0.9], dtype=np.float32),
        "overall": completion_summary(before, after),
        "clean_xyz": voxel_centers(clean),
        "faulty_xyz": voxel_centers(faulty),
        "completed_xyz": voxel_centers(completed),
        "added_xyz": voxel_centers(added),
        "true_added_xyz": voxel_centers(added),
        "false_added_xyz": np.empty((0, 3)),
        "missing_xyz": np.empty((0, 3)),
        "gate_xyz": voxel_centers(clean),
        "radar_xyz": np.array([[2.1, 0.1, 0.0]], dtype=np.float32)}
    row = _save_record(tmp_path, record, 3103)
    assert row["added_cells"] == 1
    assert (tmp_path / "test_00001.png").read_bytes().startswith(b"\x89PNG")
    assert (tmp_path / "test_00001_completed.ply").read_bytes().startswith(
        b"ply\n")
    with np.load(tmp_path / "test_00001.npz", allow_pickle=False) as archive:
        assert archive["completed_bzyx"].shape == (2, 4)
        assert archive["added_bzyx"].shape == (1, 4)
    page = (tmp_path / "test_00001.html").read_text(encoding="utf-8")
    assert "New false additions" in page
    assert "Faulty input" in page
    assert "oracle-assisted diagnostic" in page.lower()
    record["oracle_clean_latent"] = False
    _viewer(tmp_path / "deployable.html", record, 3103)
    deployable = (tmp_path / "deployable.html").read_text(encoding="utf-8")
    assert "Prediction begins from Gaussian noise" in deployable
    assert "oracle-assisted diagnostic outputs" not in deployable.lower()


def test_viewer_includes_every_point_without_display_caps(tmp_path):
    record = {
        "split": "test", "frame_id": "00001", "fault": "partial_loss_30",
        "epoch": 1, "threshold": 0.5,
        "overall": {"faulty": {"iou": 0.1, "iou_0p2m": 0.2},
                    "completed": {"iou": 0.3, "iou_0p2m": 0.4},
                    "added_cells": 1},
        "clean_xyz": np.zeros((6001, 3), dtype=np.float32),
        "faulty_xyz": np.zeros((6001, 3), dtype=np.float32),
        "true_added_xyz": np.zeros((5001, 3), dtype=np.float32),
        "false_added_xyz": np.empty((0, 3), dtype=np.float32),
        "missing_xyz": np.empty((0, 3), dtype=np.float32),
        "radar_xyz": np.zeros((2001, 3), dtype=np.float32),
        "gate_xyz": np.zeros((3001, 3), dtype=np.float32),
        "oracle_clean_latent": False,
    }
    path = tmp_path / "all_points.html"
    _viewer(path, record, 3103)
    page = path.read_text(encoding="utf-8")
    payload = json.loads(re.search(
        r'<script id="data" type="application/json">(.*?)</script>',
        page).group(1))
    assert [(layer["full"], len(layer["points"]))
            for layer in payload["layers"]] == [
                (6001, 6001), (6001, 6001), (5001, 5001), (0, 0),
                (0, 0), (2001, 2001), (3001, 3001)]
    assert "All occupancy voxel centers are displayed" in page

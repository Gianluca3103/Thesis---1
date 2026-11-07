"""Exact micro counts at every clean-denoiser occupancy threshold."""

import torch

from models.radar_lidar_stage2.clean_thresholds import (
    score_histograms, selected_indices, sweep_rows)


def test_threshold_histograms_count_boundary_scores_once():
    scores = torch.tensor([0.0, 0.5, 1.0])
    target = torch.tensor([True, False, True])
    positive, negative = score_histograms(scores, target, steps=10)
    rows = sweep_rows(positive, negative)
    assert rows[0]["tp"] == 2
    assert rows[0]["predicted"] == 3
    assert rows[5]["tp"] == 1
    assert rows[5]["predicted"] == 2
    assert rows[10]["tp"] == 1
    assert rows[10]["predicted"] == 1
    assert rows[10]["precision"] == 1.0
    assert rows[10]["recall"] == 0.5
    assert rows[10]["iou"] == 0.5


def test_constraints_are_selected_only_from_calibration():
    positive, negative = score_histograms(
        torch.tensor([0.9, 0.8, 0.4, 0.1]),
        torch.tensor([True, False, True, False]), steps=10)
    rows = sweep_rows(positive, negative)
    selected = selected_indices(rows)
    assert rows[selected["fixed_0p5"]]["threshold"] == 0.5
    assert rows[selected["best_f1"]]["f1"] == max(row["f1"] for row in rows)
    assert rows[selected["best_iou"]]["iou"] == max(row["iou"] for row in rows)
    assert rows[selected["max_precision_at_recall_90pct"]]["recall"] >= 0.9
    assert rows[selected["max_precision_at_recall_95pct"]]["recall"] >= 0.95
    assert rows[selected["max_recall_at_precision_20pct"]]["precision"] >= 0.2


def test_binned_sweep_matches_direct_thresholding():
    scores = torch.tensor([0.001, 0.101, 0.499, 0.500, 0.501,
                           0.899, 0.999], dtype=torch.float32)
    target = torch.tensor([True, False, True, False, True, False, True])
    positive, negative = score_histograms(scores, target, steps=1000)
    rows = sweep_rows(positive, negative)
    for index in (0, 1, 101, 500, 501, 899, 999, 1000):
        chosen = scores >= index / 1000
        assert rows[index]["tp"] == int((chosen & target).sum())
        assert rows[index]["predicted"] == int(chosen.sum())

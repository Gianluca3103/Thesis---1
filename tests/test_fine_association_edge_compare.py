"""The postprocessing comparison preserves the radar support contract."""

import numpy as np
import torch

from scripts.compare_fine_association_edge import (
    _add_sweep, _aggregate_sweep, _best_thresholds, _metrics,
    _near_radar_voxel, _new_sweep_totals, _print_best_results,
    _scope_masks, _sweep_frame)
from models.radar_lidar_crossmodal_v1.encoder import GRID


def test_edge_candidates_outside_three_metres_are_excluded():
    anchors = np.array([[0., 0., 0.]], dtype=np.float32)
    centers = np.array([[2.9, 0., 0.], [3.11, 0., 0.]], dtype=np.float32)
    assert _near_radar_voxel(centers, anchors, 3.0).tolist() == [True, False]


def test_equal_count_comparison_uses_only_selected_edge_cells():
    clean = torch.tensor([[0, 0, 0, 100]], dtype=torch.long)
    edge = torch.tensor([[0, 0, 0, 100], [0, 0, 0, 101]], dtype=torch.long)
    scores = np.array([0.9, 0.8], dtype=np.float32)
    full = _metrics(edge, scores, clean, 0.25)
    matched = _metrics(edge, scores, clean, 0.25, selected=np.array([0]))
    assert full["predicted"] == 2
    assert matched["predicted"] == 1
    assert matched["iou_exact"] == 1.0


def test_sweep_uses_all_clean_voxels_in_each_radar_neighborhood(capsys):
    # The clean denominator includes unproposed voxels at 3 m and in the ROI.
    clean = torch.tensor([[0, 8, 256, x] for x in (256, 261, 271, 300)])
    raw = torch.tensor([[0, 8, 256, x] for x in (256, 261)])
    edge = torch.tensor([[0, 8, 256, x] for x in (256, 261, 266)])
    origin = np.asarray(GRID.minimum_xyz)
    size = np.asarray(GRID.size_xyz)
    centers = lambda rows: origin + (rows[:, [3, 2, 1]].numpy() + .5) * size
    anchor = centers(raw[:1])
    masks = _scope_masks(centers(raw), centers(edge), centers(clean), anchor)
    scores = np.array([.9, .8])
    edge_scores = np.array([.9, .8, .7])
    frame = _sweep_frame(raw, scores, edge, edge_scores, clean, masks,
                         (.25, .85))
    assert frame["radar_1m"][.25]["raw"]["observed"] == 2
    assert frame["radar_1m"][.25]["raw"]["recall_exact"] == 1.0
    assert frame["radar_3m"][.25]["raw"]["observed"] == 3
    assert frame["radar_3m"][.25]["raw"]["recall_exact"] == 2 / 3
    assert frame["full_roi"][.25]["raw"]["observed"] == 4
    assert frame["full_roi"][.25]["raw"]["recall_exact"] == .5
    assert frame["radar_3m"][.85]["raw"]["predicted"] == 1
    assert frame["radar_3m"][.25]["edge_matched_raw_count"]["predicted"] == 2
    totals = _new_sweep_totals((.25, .85))
    _add_sweep(totals, frame)
    _add_sweep(totals, frame)
    rows = _aggregate_sweep(totals, (.25, .85))
    best = _best_thresholds(rows)
    assert best["radar_1m"]["raw"]["f1_exact"] == .25
    assert next(row for row in rows if row["scope"] == "full_roi"
                and row["variant"] == "raw"
                and row["threshold"] == .25)["observed"] == 8
    _print_best_results({"frames": 2, "split": "val", "sweep": rows,
                         "best_thresholds": best})
    printed = capsys.readouterr().out
    assert "Best thresholds by F1 over 2 val frames" in printed
    assert "radar_1m" in printed and "radar_3m" in printed
    assert "full_roi" in printed and "0.2 m" in printed

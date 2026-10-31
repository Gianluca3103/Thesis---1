"""The postprocessing comparison preserves the radar support contract."""

import numpy as np
import torch

from scripts.compare_fine_association_edge import _metrics, _near_radar_voxel


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

"""Exact and distance-tolerant occupancy IoU inside a fixed candidate gate."""

from __future__ import annotations

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import maximum_bipartite_matching
from scipy.spatial import cKDTree
import torch

from models.radar_lidar_stage1.sparse import encode_keys

from .sparse_geometry import GRID, Stage2Grid


def _centers(coords: torch.Tensor, grid: Stage2Grid) -> np.ndarray:
    ijk = coords[:, [3, 2, 1]].detach().cpu().numpy().astype(np.float64)
    return (ijk + 0.5) * np.asarray(grid.voxel_xyz) + np.asarray(grid.minimum_xyz)


def occupancy_counts(predicted: torch.Tensor, eligible_target: torch.Tensor,
                     *, tolerance_m: float = 0.2,
                     grid: Stage2Grid = GRID) -> dict[str, int]:
    """One prediction and one target can contribute to at most one match.

    Only BZYX occupancy coordinates are used. Predicted point offsets are
    intentionally excluded from both metrics.
    """
    if (predicted.ndim != 2 or predicted.shape[1] != 4
            or eligible_target.ndim != 2 or eligible_target.shape[1] != 4
            or not np.isfinite(tolerance_m) or tolerance_m <= 0):
        raise ValueError("Occupancy IoU needs BZYX cells and a positive tolerance")
    if ((len(predicted) and bool((predicted[:, 0] != 0).any()))
            or (len(eligible_target) and bool((eligible_target[:, 0] != 0).any()))):
        raise ValueError("Occupancy IoU expects one validation frame")
    predicted_keys = encode_keys(predicted.long(), grid.shape_zyx).cpu().numpy()
    target_keys = encode_keys(eligible_target.long(), grid.shape_zyx).cpu().numpy()
    if (len(np.unique(predicted_keys)) != len(predicted_keys)
            or len(np.unique(target_keys)) != len(target_keys)):
        raise ValueError("Occupancy IoU requires unique cells")
    exact = int(np.intersect1d(predicted_keys, target_keys,
                               assume_unique=True).size)
    tolerant = 0
    if len(predicted) and len(eligible_target):
        neighbors = cKDTree(_centers(eligible_target, grid)).query_ball_point(
            _centers(predicted, grid), r=tolerance_m + 1e-9, workers=1)
        lengths = np.fromiter((len(row) for row in neighbors), dtype=np.int64,
                              count=len(neighbors))
        if int(lengths.sum()):
            indptr = np.empty(len(predicted) + 1, dtype=np.int64)
            indptr[0] = 0
            np.cumsum(lengths, out=indptr[1:])
            indices = np.fromiter((index for row in neighbors for index in row),
                                  dtype=np.int32, count=int(indptr[-1]))
            graph = csr_matrix((np.ones(len(indices), dtype=np.int8), indices,
                                indptr), shape=(len(predicted), len(eligible_target)))
            tolerant = int(np.count_nonzero(
                maximum_bipartite_matching(graph, perm_type="row") >= 0))
    if not exact <= tolerant <= min(len(predicted), len(eligible_target)):
        raise RuntimeError("Distance-tolerant matching violated occupancy bounds")
    return {"predicted": len(predicted), "eligible_target": len(eligible_target),
            "exact_tp": exact, "tolerant_tp_0p2m": tolerant}


def summarize_counts(counts: dict[str, int]) -> dict[str, int | float]:
    predicted = int(counts["predicted"])
    target = int(counts["eligible_target"])
    exact = int(counts["exact_tp"])
    tolerant = int(counts["tolerant_tp_0p2m"])

    def ratio(tp: int) -> float:
        union = predicted + target - tp
        return tp / union if union else 0.0

    return {"predicted": predicted, "eligible_target": target,
            "exact_tp": exact, "tolerant_tp_0p2m": tolerant,
            "eligible_iou": ratio(exact),
            "eligible_iou_0p2m": ratio(tolerant)}

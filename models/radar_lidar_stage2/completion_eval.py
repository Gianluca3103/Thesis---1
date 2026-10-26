"""Comparable Stage-I/II scan metrics on the shared Stage-II voxel grid.

Exact and tolerant overlap use occupied voxels, not per-point multiplicity.
Tolerant matches are one-to-one between voxel centers. Chamfer uses the same
0.125 m centroid preprocessing as the existing LPCR point-cloud evaluation.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import maximum_bipartite_matching

from .lpcr_eval import metric_voxel_centroids, point_cloud_metrics
from .sparse_geometry import GRID


TOLERANCE_M = 0.2


def voxel_centers(coords_zyx: np.ndarray) -> np.ndarray:
    coords = np.asarray(coords_zyx, dtype=np.int64).reshape(-1, 3)
    minimum = np.asarray(GRID.minimum_xyz, dtype=np.float64)
    size = np.asarray(GRID.voxel_xyz, dtype=np.float64)
    return (minimum + (coords[:, ::-1] + 0.5) * size).astype(np.float32)


def scoped_cloud(points_xyz: np.ndarray, *, radar_xyz: np.ndarray | None = None,
                 radius_m: float | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Return in-ROI points and unique ZYX voxels, optionally in radar support."""
    points = np.asarray(points_xyz, dtype=np.float32).reshape(-1, 3)
    if not np.isfinite(points).all():
        raise ValueError("Cloud contains non-finite XYZ")
    lower = np.asarray(GRID.minimum_xyz, dtype=np.float32)
    upper = np.asarray(GRID.maximum_xyz, dtype=np.float32)
    points = points[((points >= lower) & (points < upper)).all(axis=1)]
    if not len(points):
        return points, np.empty((0, 3), dtype=np.int64)
    size = np.asarray(GRID.voxel_xyz, dtype=np.float32)
    ijk = np.floor((points - lower) / size).astype(np.int64)
    ijk = np.clip(ijk, 0, np.asarray(GRID.shape_zyx[::-1]) - 1)
    voxels, inverse = np.unique(ijk[:, ::-1], axis=0, return_inverse=True)
    if radar_xyz is not None:
        if radius_m is None or radius_m <= 0:
            raise ValueError("A positive radar support radius is required")
        radar = np.asarray(radar_xyz, dtype=np.float32).reshape(-1, 3)
        supported = np.zeros(len(voxels), dtype=bool)
        if len(radar):
            supported = (cKDTree(radar).query(voxel_centers(voxels), workers=1)[0]
                         <= radius_m + 1e-6)
        points = points[supported[inverse]]
        voxels = voxels[supported]
    return points, voxels


def overlap_counts(predicted_zyx: np.ndarray,
                   observed_zyx: np.ndarray) -> dict[str, int]:
    """Exact set intersection and maximum-cardinality 0.2 m voxel matching."""
    predicted = np.asarray(predicted_zyx, dtype=np.int64).reshape(-1, 3)
    observed = np.asarray(observed_zyx, dtype=np.int64).reshape(-1, 3)
    if len(np.unique(predicted, axis=0)) != len(predicted):
        raise ValueError("Predicted voxel coordinates must be unique")
    if len(np.unique(observed, axis=0)) != len(observed):
        raise ValueError("Observed voxel coordinates must be unique")
    result = {"predicted": len(predicted), "observed": len(observed),
              "tp_exact": 0, "tp_at_0_2m": 0}
    if not len(predicted) or not len(observed):
        return result
    shape = GRID.shape_zyx
    pred_keys = np.ravel_multi_index(predicted.T, shape)
    obs_keys = np.ravel_multi_index(observed.T, shape)
    result["tp_exact"] = int(np.intersect1d(pred_keys, obs_keys,
                                             assume_unique=True).size)
    edges = cKDTree(voxel_centers(predicted)).sparse_distance_matrix(
        cKDTree(voxel_centers(observed)), TOLERANCE_M + 1e-6,
        output_type="coo_matrix")
    if edges.nnz:
        graph = csr_matrix((np.ones(edges.nnz, dtype=np.int8),
                            (edges.row, edges.col)),
                           shape=(len(predicted), len(observed)))
        result["tp_at_0_2m"] = int((maximum_bipartite_matching(graph) >= 0).sum())
    if not result["tp_exact"] <= result["tp_at_0_2m"] <= min(len(predicted), len(observed)):
        raise AssertionError("Tolerant matching violated one-to-one overlap bounds")
    return result


def rates(counts: dict[str, int]) -> dict[str, float | None]:
    predicted, observed = counts["predicted"], counts["observed"]
    result: dict[str, float | None] = {}
    for suffix in ("exact", "at_0_2m"):
        tp = counts[f"tp_{suffix}"]
        union = predicted + observed - tp
        result[f"iou_{suffix}"] = tp / union if union else None
        result[f"f1_{suffix}"] = 2 * tp / (predicted + observed) if predicted + observed else None
    return result


def chamfer_squared_l2(predicted_xyz: np.ndarray,
                       observed_xyz: np.ndarray) -> float | None:
    pred = metric_voxel_centroids(predicted_xyz)
    obs = metric_voxel_centroids(observed_xyz)
    return point_cloud_metrics(pred, obs)["chamfer_squared_l2"]


def score(predicted_xyz: np.ndarray, predicted_zyx: np.ndarray,
          observed_xyz: np.ndarray, observed_zyx: np.ndarray) -> dict:
    counts = overlap_counts(predicted_zyx, observed_zyx)
    return {**counts, **rates(counts),
            "chamfer_squared_l2": chamfer_squared_l2(predicted_xyz, observed_xyz)}

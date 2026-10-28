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
FP_CATEGORIES = (
    "near_clean_0_2m",
    "known_free_before_clean_return",
    "behind_retained_faulty_return",
    "other_on_retained_return_ray",
    "no_retained_return_clean_ray",
    "no_retained_return_unmatched_ray",
)


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


def restrict_to_cells(voxels_zyx: np.ndarray,
                      cells_zyx: np.ndarray) -> np.ndarray:
    """Keep the voxels that are also members of the given cell set."""
    voxels = np.asarray(voxels_zyx, dtype=np.int64).reshape(-1, 3)
    cells = np.asarray(cells_zyx, dtype=np.int64).reshape(-1, 3)
    if not len(voxels) or not len(cells):
        return voxels[:0]
    shape = GRID.shape_zyx
    keep = np.isin(np.ravel_multi_index(voxels.T, shape),
                   np.ravel_multi_index(cells.T, shape))
    return voxels[keep]


def rates(counts: dict[str, int]) -> dict[str, float | None]:
    predicted, observed = counts["predicted"], counts["observed"]
    result: dict[str, float | None] = {}
    for suffix in ("exact", "at_0_2m"):
        tp = counts[f"tp_{suffix}"]
        union = predicted + observed - tp
        result[f"iou_{suffix}"] = tp / union if union else None
        result[f"f1_{suffix}"] = 2 * tp / (predicted + observed) if predicted + observed else None
        result[f"precision_{suffix}"] = tp / predicted if predicted else None
        result[f"recall_{suffix}"] = tp / observed if observed else None
        result[f"f0_5_{suffix}"] = (
            1.25 * tp / (predicted + 0.25 * observed)
            if predicted + observed else None)
    return result


def _first_compatible_return_range(query_xyz: np.ndarray,
                                   returns_xyz: np.ndarray, *,
                                   angular_tolerance_deg: float,
                                   lateral_tolerance_m: float) -> np.ndarray:
    """Nearest measured range along a conservative XYZ-only ray proxy."""
    query = np.asarray(query_xyz, dtype=np.float64).reshape(-1, 3)
    returns = np.asarray(returns_xyz, dtype=np.float64).reshape(-1, 3)
    first = np.full(len(query), np.inf, dtype=np.float64)
    if not len(query) or not len(returns):
        return first
    query_range = np.linalg.norm(query, axis=1)
    return_range = np.linalg.norm(returns, axis=1)
    good_query = query_range > 1e-3
    good_return = return_range > 1e-3
    if not np.any(good_query) or not np.any(good_return):
        return first
    returns = returns[good_return]
    return_range = return_range[good_return]
    query_ids = np.flatnonzero(good_query)
    max_chord = 2 * np.sin(np.deg2rad(angular_tolerance_deg) / 2)
    tree = cKDTree(returns / return_range[:, None])
    # Query in chunks to bound the temporary [candidate,16] arrays.
    for start in range(0, len(query_ids), 8192):
        ids = query_ids[start:start + 8192]
        distance, index = tree.query(
            query[ids] / query_range[ids, None],
            k=min(16, len(returns)), workers=1)
        distance = np.asarray(distance).reshape(len(ids), -1)
        index = np.asarray(index).reshape(len(ids), -1)
        compatible = ((distance <= max_chord)
                      & (query_range[ids, None] * distance <= lateral_tolerance_m))
        first[ids] = np.where(compatible, return_range[index], np.inf).min(axis=1)
    return first


def classify_addition_false_positives(
        additions_xyz: np.ndarray, clean_xyz: np.ndarray,
        faulty_xyz: np.ndarray, *, angular_tolerance_deg: float = 0.15,
        lateral_tolerance_m: float = 0.15,
        radial_margin_m: float | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Partition exact-voxel FP additions; return labels and no-retained flags.

    The XYZ-only angular match is a diagnostic proxy, not an exact beam ID.
    An unmatched clean ray is not evidence of an emitted no-return beam.
    """
    if angular_tolerance_deg <= 0 or lateral_tolerance_m <= 0:
        raise ValueError("Ray matching tolerances must be positive")
    if radial_margin_m is None:
        radial_margin_m = 0.5 * float(np.linalg.norm(GRID.voxel_xyz))
    if radial_margin_m < 0:
        raise ValueError("Radial margin must be nonnegative")
    additions, _ = scoped_cloud(additions_xyz)
    if len(additions) != len(np.asarray(additions_xyz).reshape(-1, 3)):
        raise ValueError("Generated additions must lie inside the Stage-II ROI")
    clean_returns = np.asarray(clean_xyz, dtype=np.float32).reshape(-1, 3)
    faulty_returns = np.asarray(faulty_xyz, dtype=np.float32).reshape(-1, 3)
    if not np.isfinite(clean_returns).all() or not np.isfinite(faulty_returns).all():
        raise ValueError("Clean and faulty returns must have finite XYZ")
    _, clean_voxels = scoped_cloud(clean_returns)
    labels = np.full(len(additions), "true_positive", dtype="<U40")
    no_retained = np.zeros(len(additions), dtype=bool)
    if not len(additions):
        return labels, no_retained
    xyz_min = np.asarray(GRID.minimum_xyz, dtype=np.float64)
    size = np.asarray(GRID.voxel_xyz, dtype=np.float64)
    addition_ijk = np.floor((additions.astype(np.float64) - xyz_min) / size).astype(np.int64)
    addition_keys = np.ravel_multi_index(addition_ijk[:, ::-1].T, GRID.shape_zyx)
    clean_keys = np.ravel_multi_index(clean_voxels.T, GRID.shape_zyx)
    false_index = np.flatnonzero(~np.isin(addition_keys, clean_keys))
    if not len(false_index):
        return labels, no_retained
    false_xyz = additions[false_index]
    clean_range = _first_compatible_return_range(
        false_xyz, clean_returns, angular_tolerance_deg=angular_tolerance_deg,
        lateral_tolerance_m=lateral_tolerance_m)
    faulty_range = _first_compatible_return_range(
        false_xyz, faulty_returns, angular_tolerance_deg=angular_tolerance_deg,
        lateral_tolerance_m=lateral_tolerance_m)
    no_retained[false_index] = ~np.isfinite(faulty_range)
    query_range = np.linalg.norm(false_xyz, axis=1)
    near_clean = (cKDTree(clean_returns).query(false_xyz, workers=1)[0]
                  <= TOLERANCE_M + 1e-6 if len(clean_returns)
                  else np.zeros(len(false_xyz), dtype=bool))
    categories = np.full(len(false_index), "no_retained_return_unmatched_ray", dtype="<U40")
    categories[np.isfinite(clean_range)] = "no_retained_return_clean_ray"
    categories[np.isfinite(faulty_range)] = "other_on_retained_return_ray"
    categories[query_range > faulty_range + radial_margin_m] = "behind_retained_faulty_return"
    categories[np.isfinite(clean_range)
               & (query_range + radial_margin_m < clean_range)] = (
                   "known_free_before_clean_return")
    categories[near_clean] = "near_clean_0_2m"
    labels[false_index] = categories
    return labels, no_retained


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

"""Optional radar-feature-guided interpolation of raw surface proposals.

This is an inference-only comparison. It does not change the trained student's
raw locations or scores and never receives clean LiDAR.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree

from .encoder import GRID


@dataclass(frozen=True)
class EdgeAwareField:
    coordinates_xyz: np.ndarray  # fine grid XYZ indices
    scores: np.ndarray
    batch_index: np.ndarray
    source_seeds: int
    accepted_edges: int

    def centers_xyz(self) -> np.ndarray:
        origin = np.asarray(GRID.minimum_xyz, dtype=np.float64)
        size = np.asarray(GRID.size_xyz, dtype=np.float64)
        return (origin + (self.coordinates_xyz + .5) * size).astype(np.float32)


def _compact_splat(xyz: np.ndarray, scores: np.ndarray,
                   lattice_stride: int) -> tuple[np.ndarray, np.ndarray]:
    """Local hat basis around accepted seeds and graph links, without a sphere."""
    shape = np.asarray(GRID.shape_zyx[::-1], dtype=np.int64)
    step = np.asarray(GRID.size_xyz, dtype=np.float64)
    origin = np.asarray(GRID.minimum_xyz, dtype=np.float64)
    keys, values = [], []
    for center_xyz, score in zip(xyz, scores):
        center = (center_xyz - origin) / step - .5
        lo = np.maximum(0, np.floor(center - lattice_stride).astype(np.int64) + 1)
        hi = np.minimum(shape - 1, np.ceil(center + lattice_stride).astype(np.int64) - 1)
        if np.any(lo > hi):
            continue
        cells = np.stack(np.meshgrid(*(np.arange(a, b + 1) for a, b in zip(lo, hi)),
                                     indexing="ij"), axis=-1).reshape(-1, 3)
        relative = np.abs((cells - center) / lattice_stride)
        weight = np.prod(np.maximum(0., 1. - relative), axis=1)
        positive = weight > 0
        if not positive.any():
            continue
        cells, weight = cells[positive], weight[positive]
        weight = weight / weight.max()
        keys.append(np.ravel_multi_index(cells.T, shape))
        values.append(score * weight)
    if not keys:
        return np.empty((0, 3), dtype=np.int32), np.empty(0, dtype=np.float32)
    flat_keys = np.concatenate(keys)
    flat_values = np.concatenate(values)
    order = np.argsort(flat_keys)
    sorted_keys = flat_keys[order]
    first = np.flatnonzero(np.r_[True, sorted_keys[1:] != sorted_keys[:-1]])
    unique = sorted_keys[first]
    maximum = np.maximum.reduceat(flat_values[order], first)
    cells = np.stack(np.unravel_index(unique, tuple(shape)), axis=1).astype(np.int32)
    return cells, maximum.astype(np.float32)


def edge_aware_surface(xyz: np.ndarray, scores: np.ndarray,
                       radar_features: np.ndarray, anchor_group: np.ndarray,
                       batch_index: np.ndarray | None = None, *,
                       seed_threshold: float = .25, lattice_stride: int = 2,
                       max_link_m: float = 2., feature_sigma: float = .25,
                       score_sigma: float = .20, min_affinity: float = .20,
                       neighbors: int = 8) -> EdgeAwareField:
    xyz = np.asarray(xyz, dtype=np.float64)
    scores = np.asarray(scores, dtype=np.float64)
    features = np.asarray(radar_features, dtype=np.float64)
    group = np.asarray(anchor_group)
    batch = (np.zeros(len(xyz), dtype=np.int64) if batch_index is None
             else np.asarray(batch_index))
    if (xyz.ndim != 2 or xyz.shape[1] != 3 or scores.shape != (len(xyz),)
            or features.ndim != 2 or len(features) != len(xyz)
            or group.shape != (len(xyz),) or batch.shape != (len(xyz),)
            or not np.isfinite(xyz).all() or not np.isfinite(scores).all()
            or not np.isfinite(features).all() or np.any((scores < 0) | (scores > 1))
            or not 0 <= seed_threshold <= 1 or lattice_stride < 1
            or min(max_link_m, feature_sigma, score_sigma) <= 0
            or not 0 < min_affinity < 1 or neighbors < 1):
        raise ValueError("Invalid raw surface proposals or edge-aware settings")
    chosen = scores > seed_threshold
    xyz, scores, features, group, batch = (item[chosen] for item in
                                             (xyz, scores, features, group, batch))
    if not len(xyz):
        empty = np.empty((0, 3), dtype=np.int32)
        return EdgeAwareField(empty, np.empty(0, dtype=np.float32),
                              np.empty(0, dtype=np.int64), 0, 0)
    features = features / np.maximum(np.linalg.norm(features, axis=1, keepdims=True), 1e-12)
    edges: dict[tuple[int, int], float] = {}
    for batch_id in np.unique(batch):
        local_index = np.flatnonzero(batch == batch_id)
        if len(local_index) < 2:
            continue
        distance, local = cKDTree(xyz[local_index]).query(
            xyz[local_index], k=min(neighbors + 1, len(local_index)),
            distance_upper_bound=max_link_m)
        distance = np.asarray(distance).reshape(len(local_index), -1)
        local = np.asarray(local).reshape(len(local_index), -1)
        for row, i in enumerate(local_index):
            for d, column in zip(distance[row, 1:], local[row, 1:]):
                if column >= len(local_index):
                    continue
                j = int(local_index[column])
                if i == j or group[i] == group[j]:
                    continue
                cosine = float(np.clip(features[i] @ features[j], -1, 1))
                affinity = np.exp(-(1 - cosine) / feature_sigma
                                  - .5 * ((scores[i] - scores[j]) / score_sigma) ** 2
                                  - .5 * (d / max_link_m) ** 2)
                if affinity >= min_affinity:
                    key = (min(i, j), max(i, j))
                    edges[key] = max(edges.get(key, 0.), float(affinity))
    smoothed = scores.copy()
    if edges:
        sums = scores.copy()
        weights = np.ones(len(scores), dtype=np.float64)
        for (i, j), affinity in edges.items():
            sums[i] += affinity * scores[j]
            sums[j] += affinity * scores[i]
            weights[i] += affinity
            weights[j] += affinity
        smoothed = sums / weights
    all_xyz, all_scores, all_batch = [xyz], [smoothed], [batch]
    step = min(GRID.size_xyz)
    for (i, j), affinity in edges.items():
        count = max(1, int(np.ceil(np.linalg.norm(xyz[j] - xyz[i]) / step)))
        fraction = np.arange(1, count) / count
        if len(fraction):
            all_xyz.append(xyz[i] + fraction[:, None] * (xyz[j] - xyz[i]))
            all_scores.append((smoothed[i] * (1 - fraction) + smoothed[j] * fraction)
                              * affinity)
            all_batch.append(np.full(len(fraction), batch[i], dtype=np.int64))
    all_xyz = np.concatenate(all_xyz)
    all_scores = np.concatenate(all_scores)
    all_batch = np.concatenate(all_batch)
    coordinates, values, batches = [], [], []
    for batch_id in np.unique(all_batch):
        selected = all_batch == batch_id
        sites, scores_b = _compact_splat(all_xyz[selected], all_scores[selected],
                                        lattice_stride)
        coordinates.append(sites)
        values.append(scores_b)
        batches.append(np.full(len(sites), batch_id, dtype=np.int64))
    return EdgeAwareField(np.concatenate(coordinates), np.concatenate(values),
                          np.concatenate(batches), len(xyz), len(edges))

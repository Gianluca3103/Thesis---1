"""Clean-LiDAR occupancy and centroid targets on a fixed candidate domain."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree
import torch

from models.radar_lidar_stage1.sparse import encode_keys, voxelize
from .candidate_domain import CandidateDomain, voxel_centers_xyz
from .cpu_query import query_workers


@dataclass(frozen=True)
class VoxelTargets:
    occupied: torch.Tensor  # [N] positive if a measured clean point is in voxel
    known_free: torch.Tensor  # [N] conservatively observed before a clean first return
    offsets_normalized: torch.Tensor  # [N,3], meaningful only for positives
    clean_centroid_xyz: torch.Tensor  # [N,3], meaningful only for positives
    clean_point_count: torch.Tensor  # [N]
    clean_points_in_grid: int
    clean_points_in_candidates: int
    point_centroid_errors_m: torch.Tensor  # per represented clean return

    @property
    def candidate_occupancy_ratio(self) -> float:
        return float(self.occupied.float().mean()) if len(self.occupied) else 0.0

    @property
    def clean_point_coverage(self) -> float:
        return self.clean_points_in_candidates / max(self.clean_points_in_grid, 1)


def make_targets(domain: CandidateDomain, clean_lidar: torch.Tensor,
                 clean_valid: torch.Tensor, *, free_ray_tolerance_m: float = 0.15) -> VoxelTargets:
    """Measure clean geometry in the candidate voxels; no target alters support.

    An unoccupied candidate is not necessarily physically free: occluded and
    unobserved sites require a separate visibility mask before an occupancy
    loss may label them negative.
    """
    grid = domain.grid
    if clean_lidar.device != domain.coordinates.device:
        raise ValueError("Clean LiDAR and candidate coordinates must share a device")
    clean_coords, inverse, selected, _ = voxelize(clean_lidar, clean_valid, grid)
    keys = encode_keys(domain.coordinates, grid.shape_zyx)
    clean_keys = encode_keys(clean_coords, grid.shape_zyx)
    n = len(keys)
    occupied = torch.zeros(n, dtype=torch.bool, device=keys.device)
    known_free = torch.zeros_like(occupied)
    counts = torch.zeros(n, dtype=torch.long, device=keys.device)
    centroids = clean_lidar.new_zeros((n, 3))
    normalized = clean_lidar.new_zeros((n, 3))
    if not n or not len(clean_keys):
        return VoxelTargets(occupied, known_free, normalized, centroids, counts, len(selected), 0,
                            clean_lidar.new_empty(0))
    position = torch.searchsorted(keys, clean_keys)
    valid = position < n
    matched = valid & (keys[position.clamp(max=n - 1)] == clean_keys)
    # First pool measured points by clean voxel, then transfer only matching
    # voxels into the candidate domain. This preserves the exact centroid.
    # Summing absolute XYZ in float32 loses precision for dense, distant
    # voxels. Accumulate offsets from each cell center in float64 instead.
    clean_centers = voxel_centers_xyz(clean_coords, grid, dtype=torch.float64)
    voxel_sum = torch.zeros((len(clean_keys), 3), dtype=torch.float64, device=keys.device)
    voxel_sum.index_add_(0, inverse, selected[:, :3].double() - clean_centers[inverse])
    voxel_count = torch.zeros(len(clean_keys), dtype=torch.long, device=keys.device)
    voxel_count.index_add_(0, inverse, torch.ones_like(inverse))
    centroid_offset = voxel_sum / voxel_count.clamp_min(1)[:, None]
    centroid_by_clean = clean_centers + centroid_offset
    dest = position[matched]
    occupied[dest] = True
    counts[dest] = voxel_count[matched]
    centroids[dest] = centroid_by_clean[matched].to(centroids.dtype)
    sizes = torch.tensor(grid.size_xyz, dtype=torch.float64, device=keys.device)
    clean_normalized = centroid_offset[matched] / sizes
    # Stage I bins points in float32. A point within a few micrometres of a
    # mathematical boundary can round into its adjacent cell. Project only
    # that numerical sliver onto the representable cell edge.
    excess = clean_normalized.abs() - 0.5
    if bool((excess > 1e-4).any()):
        raise AssertionError(f"Clean voxel centroid escaped its physical cell by "
                             f"{float(excess.max()):.6g} normalized voxels")
    normalized[dest] = clean_normalized.clamp(-0.5, 0.5).to(normalized.dtype)
    represented = matched[inverse]
    point_errors = torch.linalg.vector_norm(
        selected[represented, :3] - centroid_by_clean[inverse[represented]].to(selected.dtype), dim=-1)
    # Absence of a measured point is *unknown*. A negative is justified only
    # where a clean scan ray passed through this cell before its first return.
    # The sensor origin is (0,0,0) in calibrated LiDAR coordinates. We require
    # a tight perpendicular distance to an actual measured ray and keep one
    # half voxel diagonal clear of its endpoint; no space behind is negative.
    centers = domain.centers_xyz_numpy()
    batches = domain.batch_numpy()
    radius = np.linalg.norm(centers, axis=1)
    half_diagonal = 0.5 * float(np.linalg.norm(grid.size_xyz))
    for batch in np.unique(batches):
        candidate_index = np.flatnonzero(batches == batch)
        measured = clean_lidar[int(batch), clean_valid[int(batch)], :3].detach().cpu().numpy()
        measured_range = np.linalg.norm(measured, axis=1)
        good = measured_range > 1e-3
        if not np.any(good):
            continue
        measured, measured_range = measured[good], measured_range[good]
        ray = measured / measured_range[:, None]
        query = centers[candidate_index]
        query_range = radius[candidate_index]
        nonzero = query_range > 1e-3
        if not np.any(nonzero):
            continue
        # Several measured returns can have nearly the same direction. Use
        # the earliest compatible return; choosing an arbitrary farther ray
        # could falsely mark space behind a nearer occluder as free.
        chord, nearest = cKDTree(ray).query(query[nonzero] / query_range[nonzero, None],
                                                k=min(16, len(ray)),
                                                workers=query_workers(int(nonzero.sum())))
        chord = np.atleast_2d(chord).reshape(np.count_nonzero(nonzero), -1)
        nearest = np.atleast_2d(nearest).reshape(np.count_nonzero(nonzero), -1)
        compatible = query_range[nonzero, None] * chord <= free_ray_tolerance_m
        first_range = np.where(compatible, measured_range[nearest], np.inf).min(axis=1)
        free = query_range[nonzero] + half_diagonal < first_range
        free &= np.isfinite(first_range)
        selected_candidates = candidate_index[nonzero][free]
        known_free[selected_candidates] = True
    known_free &= ~occupied
    return VoxelTargets(occupied, known_free, normalized, centroids, counts,
                        len(selected), int(voxel_count[matched].sum()), point_errors)


def decode_centroids(domain: CandidateDomain, offsets_normalized: torch.Tensor) -> torch.Tensor:
    if offsets_normalized.shape != (len(domain.coordinates), 3):
        raise ValueError("Expected one normalized XYZ offset per candidate voxel")
    sizes = offsets_normalized.new_tensor(domain.grid.size_xyz)
    return domain.centers_xyz + offsets_normalized.clamp(-0.5, 0.5) * sizes

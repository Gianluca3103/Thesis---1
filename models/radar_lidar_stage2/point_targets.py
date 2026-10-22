"""Multi-point supervision derived from clean XYZ without changing fixed caches."""

from __future__ import annotations

import torch

from models.radar_lidar_stage1.sparse import encode_keys

from .sparse_geometry import GRID, Stage2Grid


def multipoint_targets(clean_xyz: torch.Tensor, allowed: torch.Tensor,
                       point_batch_ids: torch.Tensor | None = None,
                       *, max_points: int = 5, grid: Stage2Grid = GRID
                       ) -> tuple[torch.Tensor, torch.Tensor]:
    """Measured point counts and up to five center-relative XYZ offsets on A.

    Dense voxels are subsampled evenly in the original scan order. The
    decoder loss matches these points as a set, so target order is immaterial.
    """
    if clean_xyz.ndim != 2 or clean_xyz.shape[1] != 3 or max_points < 1:
        raise ValueError("Expected XYZ points and a positive point cap")
    if point_batch_ids is None:
        point_batch_ids = torch.zeros(len(clean_xyz), dtype=torch.long,
                                      device=clean_xyz.device)
    if point_batch_ids.shape != (len(clean_xyz),):
        raise ValueError("Expected one batch ID per clean point")
    counts = torch.zeros(len(allowed), dtype=torch.long, device=allowed.device)
    offsets = clean_xyz.new_zeros((len(allowed), max_points, 3))
    if not len(clean_xyz) or not len(allowed):
        return counts, offsets
    minimum = clean_xyz.new_tensor(grid.minimum_xyz)
    size = clean_xyz.new_tensor(grid.voxel_xyz)
    index = torch.floor((clean_xyz - minimum) / size).long()
    inside = ((index >= 0) & (index < index.new_tensor(grid.shape_zyx[::-1]))).all(1)
    if not bool(inside.any()):
        return counts, offsets
    point_coords = torch.stack((point_batch_ids[inside].long(),
                                index[inside, 2], index[inside, 1],
                                index[inside, 0]), dim=1)
    point_keys = encode_keys(point_coords, grid.shape_zyx)
    allowed_keys = encode_keys(allowed.long(), grid.shape_zyx)
    site = torch.searchsorted(allowed_keys, point_keys)
    matched = (site < len(allowed_keys)) & (allowed_keys[
        site.clamp(max=len(allowed_keys) - 1)] == point_keys)
    site = site[matched]
    xyz = clean_xyz[inside][matched]
    if not len(site):
        return counts, offsets
    order = torch.argsort(site, stable=True)
    site, xyz = site[order], xyz[order]
    counts = torch.bincount(site, minlength=len(allowed))
    kept = counts.clamp(max=max_points)
    starts = counts.cumsum(0) - counts
    centers = minimum + (allowed[:, [3, 2, 1]].to(xyz.dtype) + 0.5) * size
    for slot in range(max_points):
        present = kept > slot
        # Spread five representatives over a dense voxel without randomness.
        position = (starts[present] + torch.div(
            slot * (counts[present] - 1),
            (kept[present] - 1).clamp_min(1), rounding_mode="floor"))
        offsets[present, slot] = (xyz[position] - centers[present]).clamp(
            min=-size * 0.5, max=size * 0.5)
    return kept, offsets

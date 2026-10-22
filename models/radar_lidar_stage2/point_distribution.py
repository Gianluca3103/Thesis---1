"""Measured point multiplicity and within-voxel coordinates on Stage-I sites."""

from __future__ import annotations

import torch

from models.radar_lidar_stage1.sparse import encode_keys

from .sparse_geometry import GRID, Stage2Grid


def points_on_allowed(clean_xyz: torch.Tensor, allowed: torch.Tensor,
                      point_batch_ids: torch.Tensor | None = None,
                      *, grid: Stage2Grid = GRID
                      ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return allowed-site indices, local XYZ in [-1,1], and counts per site.

    The site set is supplied by Stage I. Clean points only populate targets.
    """
    if clean_xyz.ndim != 2 or clean_xyz.shape[1] != 3:
        raise ValueError("Clean LiDAR XYZ must be [N,3]")
    if allowed.ndim != 2 or allowed.shape[1] != 4:
        raise ValueError("Allowed coordinates must be [S,4]")
    if point_batch_ids is None:
        point_batch_ids = torch.zeros(len(clean_xyz), dtype=torch.long,
                                      device=clean_xyz.device)
    if (point_batch_ids.shape != (len(clean_xyz),)
            or point_batch_ids.device != clean_xyz.device
            or point_batch_ids.dtype not in (torch.int32, torch.int64)
            or allowed.device != clean_xyz.device):
        raise ValueError("Point batch IDs and allowed sites must share the point device")
    counts = torch.zeros(len(allowed), dtype=torch.long, device=clean_xyz.device)
    if not len(clean_xyz) or not len(allowed):
        return (counts.new_empty(0), clean_xyz.new_empty((0, 3)), counts)
    minimum = clean_xyz.new_tensor(grid.minimum_xyz)
    maximum = clean_xyz.new_tensor(grid.maximum_xyz)
    voxel = clean_xyz.new_tensor(grid.voxel_xyz)
    inside = ((clean_xyz >= minimum) & (clean_xyz < maximum)).all(1)
    xyz = clean_xyz[inside]
    batches = point_batch_ids[inside].long()
    if not len(xyz):
        return (counts.new_empty(0), clean_xyz.new_empty((0, 3)), counts)
    ijk = torch.floor((xyz - minimum) / voxel).long()
    coords = torch.cat((batches[:, None], ijk[:, [2, 1, 0]]), dim=1)
    keys = encode_keys(allowed.long(), grid.shape_zyx)
    positions = torch.searchsorted(keys, encode_keys(coords, grid.shape_zyx))
    matched = positions < len(keys)
    matched &= keys[positions.clamp(max=len(keys) - 1)] == encode_keys(
        coords, grid.shape_zyx)
    site_index = positions[matched]
    xyz = xyz[matched]
    ijk = ijk[matched]
    counts = torch.bincount(site_index, minlength=len(allowed))
    centers = minimum + (ijk.to(xyz.dtype) + 0.5) * voxel
    local = ((xyz - centers) / (voxel * 0.5)).clamp(-1, 1)
    return site_index, local, counts

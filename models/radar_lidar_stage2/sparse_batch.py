"""Batch-aware LiDAR voxelization outside the fixed-cache geometry contract."""

from __future__ import annotations

import torch

from models.radar_lidar_stage1.sparse import encode_keys

from .sparse_geometry import GRID, Stage2Grid


def voxelize_means_batched(points_xyz: torch.Tensor,
                           point_batch_ids: torch.Tensor,
                           *, grid: Stage2Grid = GRID
                           ) -> tuple[torch.Tensor, torch.Tensor]:
    """Mean XYZ per occupied BZYX voxel, keeping overlapping frames separate."""
    xyz = points_xyz[:, :3]
    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError("LiDAR XYZ must be [N,3]")
    if (point_batch_ids.shape != (len(xyz),) or
            point_batch_ids.device != xyz.device or
            point_batch_ids.dtype not in (torch.int32, torch.int64)):
        raise ValueError("Point batch IDs must be an integer [N] tensor on the point device")
    minimum = xyz.new_tensor(grid.minimum_xyz)
    maximum = xyz.new_tensor(grid.maximum_xyz)
    size = xyz.new_tensor(grid.voxel_xyz)
    inside = ((xyz >= minimum) & (xyz < maximum)).all(-1)
    batch_ids = point_batch_ids[inside].long()
    xyz = xyz[inside]
    if not len(xyz):
        return (torch.empty((0, 4), dtype=torch.long, device=points_xyz.device),
                points_xyz.new_empty((0, 3)))
    ijk = torch.floor((xyz - minimum) / size).long()
    coords = torch.cat((batch_ids[:, None], ijk[:, [2, 1, 0]]), dim=1)
    keys = encode_keys(coords, grid.shape_zyx)
    unique, inverse = torch.unique(keys, sorted=True, return_inverse=True)
    sums = xyz.new_zeros((len(unique), 3))
    sums.index_add_(0, inverse, xyz)
    counts = torch.bincount(inverse, minlength=len(unique)).to(xyz.dtype)
    z, y, x = grid.shape_zyx
    xx = unique % x
    rest = unique // x
    yy = rest % y
    rest = rest // y
    zz = rest % z
    batch = rest // z
    return torch.stack((batch, zz, yy, xx), dim=1), sums / counts[:, None]

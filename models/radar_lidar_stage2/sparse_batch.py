"""Batch-aware LiDAR voxelization outside the fixed-cache geometry contract."""

from __future__ import annotations

import torch

from models.radar_lidar_stage1.sparse import encode_keys

from .sparse_geometry import GRID, Stage2Grid


def voxelize_lidar_features(points: torch.Tensor,
                            point_batch_ids: torch.Tensor | None = None,
                            *, grid: Stage2Grid = GRID
                            ) -> tuple[torch.Tensor, torch.Tensor]:
    """Mean XYZ/intensity, local spread and count per occupied LiDAR voxel."""
    if points.ndim != 2 or points.shape[1] != 4:
        raise ValueError("Shared LiDAR encoder expects XYZ and reflectivity [N,4]")
    if point_batch_ids is None:
        point_batch_ids = torch.zeros(len(points), dtype=torch.long,
                                      device=points.device)
    if (point_batch_ids.shape != (len(points),)
            or point_batch_ids.device != points.device
            or point_batch_ids.dtype not in (torch.int32, torch.int64)):
        raise ValueError("LiDAR batch IDs must be integer [N] on the point device")
    xyz = points[:, :3]
    minimum = xyz.new_tensor(grid.minimum_xyz)
    maximum = xyz.new_tensor(grid.maximum_xyz)
    size = xyz.new_tensor(grid.voxel_xyz)
    inside = ((xyz >= minimum) & (xyz < maximum)).all(-1)
    xyz = xyz[inside]
    batch_ids = point_batch_ids[inside].long()
    if not len(xyz):
        return (torch.empty((0, 4), dtype=torch.long, device=points.device),
                points.new_empty((0, 8)))
    ijk = torch.floor((xyz - minimum) / size).long()
    coords = torch.cat((batch_ids[:, None], ijk[:, [2, 1, 0]]), dim=1)
    unique, inverse = torch.unique(encode_keys(coords, grid.shape_zyx),
                                   sorted=True, return_inverse=True)
    values = torch.cat((xyz, points[inside, 3:4] / 255.0), dim=1)
    sums = points.new_zeros((len(unique), 4))
    sums.index_add_(0, inverse, values)
    counts = torch.bincount(inverse, minlength=len(unique)).to(points.dtype)
    centers = minimum + (ijk.to(points.dtype) + 0.5) * size
    local = (xyz - centers) / (size * 0.5)
    local_sum = points.new_zeros((len(unique), 3))
    local_square_sum = points.new_zeros((len(unique), 3))
    local_sum.index_add_(0, inverse, local)
    local_square_sum.index_add_(0, inverse, local.square())
    local_mean = local_sum / counts[:, None]
    spread = (local_square_sum / counts[:, None] - local_mean.square()
              ).clamp_min(0).sqrt()
    z, y, x = grid.shape_zyx
    xx = unique % x
    rest = unique // x
    yy = rest % y
    rest //= y
    zz = rest % z
    batch = rest // z
    features = torch.cat((sums / counts[:, None], spread,
                          torch.log1p(counts)[:, None]), dim=1)
    return torch.stack((batch, zz, yy, xx), dim=1), features


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

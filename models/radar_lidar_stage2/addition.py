"""Faulty-LiDAR context and supervision for additive Stage-II completion.

Clean LiDAR is used only to label training sites. Inference features come from
the faulty scan and the frame-specific radar gate.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from models.radar_lidar_stage1.sparse import decode_keys, encode_keys

from .sparse_geometry import GRID, Stage2Grid


REGION_STRIDE_ZYX = (4, 4, 4)  # 0.8 x 0.8 x 1.0 m
MIN_CLEAN_CELLS_IN_REGION = 8
HEALTHY_RETAINED_FRACTION = 0.85


@dataclass(frozen=True)
class AdditionContext:
    observed: torch.Tensor
    local_faulty_density: torch.Tensor
    region_keys: torch.Tensor
    observed_coords: torch.Tensor


def _voxel_coords(points_xyz: torch.Tensor,
                  point_batch_ids: torch.Tensor | None,
                  grid: Stage2Grid) -> torch.Tensor:
    # Only occupancy is needed. Avoid the centroid aggregation performed by
    # the encoder's voxelizer on every diffusion training step.
    if points_xyz.ndim != 2 or points_xyz.shape[1] != 3:
        raise ValueError("Addition context requires LiDAR XYZ rows")
    if point_batch_ids is None:
        point_batch_ids = torch.zeros(len(points_xyz), dtype=torch.long,
                                      device=points_xyz.device)
    if (point_batch_ids.shape != (len(points_xyz),)
            or point_batch_ids.device != points_xyz.device
            or point_batch_ids.dtype not in (torch.int32, torch.int64)):
        raise ValueError("Addition context batch IDs differ from LiDAR points")
    minimum = points_xyz.new_tensor(grid.minimum_xyz)
    maximum = points_xyz.new_tensor(grid.maximum_xyz)
    inside = ((points_xyz >= minimum) & (points_xyz < maximum)).all(1)
    xyz = points_xyz[inside]
    if not len(xyz):
        return torch.empty((0, 4), dtype=torch.long, device=points_xyz.device)
    index = torch.floor((xyz - minimum) /
                        xyz.new_tensor(grid.voxel_xyz)).long()
    coords = torch.cat((point_batch_ids[inside].long()[:, None],
                        index[:, [2, 1, 0]]), dim=1)
    return decode_keys(torch.unique(encode_keys(coords, grid.shape_zyx),
                                    sorted=True), grid.shape_zyx)


def _region_shape(grid: Stage2Grid) -> tuple[int, int, int]:
    return tuple((size + stride - 1) // stride for size, stride in
                 zip(grid.shape_zyx, REGION_STRIDE_ZYX))


def _region_keys(coords: torch.Tensor, grid: Stage2Grid) -> torch.Tensor:
    region = coords.long().clone()
    region[:, 1:] //= region.new_tensor(REGION_STRIDE_ZYX)
    return encode_keys(region, _region_shape(grid))


@torch.no_grad()
def addition_context(faulty_xyz: torch.Tensor, allowed: torch.Tensor,
                     point_batch_ids: torch.Tensor | None = None,
                     *, grid: Stage2Grid = GRID) -> AdditionContext:
    """Return observed cells and region density without reading clean LiDAR."""
    observed_coords = _voxel_coords(faulty_xyz, point_batch_ids, grid)
    observed = torch.isin(encode_keys(allowed, grid.shape_zyx),
                          encode_keys(observed_coords, grid.shape_zyx))
    region_keys = _region_keys(allowed, grid)
    shape = _region_shape(grid)
    batch_count = int(allowed[:, 0].max()) + 1 if len(allowed) else 1
    region_count = batch_count * shape[0] * shape[1] * shape[2]
    counts = torch.bincount(_region_keys(observed_coords, grid),
                            minlength=region_count)
    # Bounded feature; the head also receives LPCR geometry and radar score.
    local_density = torch.log1p(counts[region_keys].float()) / 4.0
    return AdditionContext(observed, local_density, region_keys,
                           observed_coords)


@torch.no_grad()
def healthy_region_negatives(clean_xyz: torch.Tensor,
                             allowed: torch.Tensor,
                             clean_occupied: torch.Tensor,
                             context: AdditionContext,
                             clean_batch_ids: torch.Tensor | None = None,
                             *, grid: Stage2Grid = GRID) -> torch.Tensor:
    """Label empty sites in regions where most clean cells were retained.

    This oracle is only a training loss mask. A genuinely missing clean cell
    remains positive even when other cells in its region are healthy.
    """
    clean_coords = _voxel_coords(clean_xyz, clean_batch_ids, grid)
    shape = _region_shape(grid)
    batch_count = int(allowed[:, 0].max()) + 1 if len(allowed) else 1
    region_count = batch_count * shape[0] * shape[1] * shape[2]
    clean_counts = torch.bincount(_region_keys(clean_coords, grid),
                                  minlength=region_count)
    retained = torch.isin(encode_keys(clean_coords, grid.shape_zyx),
                          encode_keys(context.observed_coords, grid.shape_zyx))
    retained_counts = torch.bincount(
        _region_keys(clean_coords[retained], grid), minlength=region_count)
    healthy = ((clean_counts >= MIN_CLEAN_CELLS_IN_REGION)
               & (retained_counts.float() >=
                  HEALTHY_RETAINED_FRACTION * clean_counts.float()))
    return (healthy[context.region_keys] & ~clean_occupied & ~context.observed)

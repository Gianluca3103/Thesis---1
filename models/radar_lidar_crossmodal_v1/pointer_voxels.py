"""Merge radar-only surface pointers into unique fine-resolution LiDAR voxels."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from models.radar_lidar_stage1.sparse import decode_keys, encode_keys

from .encoder import GRID


@dataclass
class PointerVoxelField:
    coords: torch.Tensor  # sorted unique [N, batch,z,y,x]
    confidence: torch.Tensor  # one confidence per unique voxel
    source_count: torch.Tensor  # distinct radar anchors pointing to each voxel
    representative_source: torch.Tensor  # strongest pointer's radar anchor
    pointer_xyz: torch.Tensor  # in-grid continuous pointer coordinates
    pointer_batch: torch.Tensor
    pointer_strength: torch.Tensor
    pointer_voxel_index: torch.Tensor  # pointer -> unique voxel, for training gradients


def merge_surface_pointers(proposals) -> PointerVoxelField:
    """Deduplicate slots per radar anchor, then combine independent anchors.

    All slots participate until the *merged* confidence is thresholded at
    inference. Thus several individually weak pointers may jointly support a
    voxel. No LiDAR data and no neighboring output cells are used here.
    """
    xyz = proposals.xyz.reshape(-1, 3)
    strength = proposals.score.reshape(-1)
    slots = proposals.xyz.shape[1]
    source = torch.arange(len(proposals.anchor_xyz), device=xyz.device).repeat_interleave(slots)
    batch = proposals.anchor_batch.repeat_interleave(slots)
    origin = xyz.new_tensor(GRID.minimum_xyz)
    size = xyz.new_tensor(GRID.size_xyz)
    cell_xyz = torch.floor((xyz - origin) / size).long()
    shape_xyz = cell_xyz.new_tensor(GRID.shape_zyx[::-1])
    inside = torch.isfinite(xyz).all(-1) & ((cell_xyz >= 0) & (cell_xyz < shape_xyz)).all(-1)
    if proposals.max_voxel_center_distance_m is not None:
        # The continuous pointer can be inside the support ball while its
        # destination voxel center lies just outside it. Association is to
        # the voxel, so enforce the radius on that center too.
        voxel_centers = origin + (cell_xyz.to(xyz.dtype) + 0.5) * size
        distance = torch.linalg.vector_norm(
            voxel_centers - proposals.anchor_xyz[source], dim=-1)
        inside &= distance <= proposals.max_voxel_center_distance_m + 1e-6
    xyz, strength, source, batch, cell_xyz = (
        item[inside] for item in (xyz, strength, source, batch, cell_xyz))
    if not len(xyz):
        empty_coords = batch.new_empty((0, 4))
        empty_long = batch.new_empty((0,))
        return PointerVoxelField(empty_coords, strength, empty_long, empty_long,
                                 xyz, batch, strength, empty_long)

    rows = torch.stack((batch, cell_xyz[:, 2], cell_xyz[:, 1], cell_xyz[:, 0]), -1)
    keys = encode_keys(rows, GRID.shape_zyx)
    volume = GRID.shape_zyx[0] * GRID.shape_zyx[1] * GRID.shape_zyx[2]
    total_keys = volume * (int(batch.max()) + 1)
    source_keys, source_inverse = torch.unique(source * total_keys + keys,
                                               sorted=True, return_inverse=True)
    per_source = strength.new_zeros(len(source_keys))
    per_source.scatter_reduce_(0, source_inverse, strength, reduce="amax", include_self=True)
    voxel_keys, voxel_inverse = torch.unique(source_keys % total_keys,
                                             sorted=True, return_inverse=True)
    log_absence = torch.log1p(-per_source.clamp(max=1 - 1e-6))
    summed = strength.new_zeros(len(voxel_keys)).index_add(0, voxel_inverse, log_absence)
    confidence = -torch.expm1(summed)
    source_count = torch.bincount(voxel_inverse, minlength=len(voxel_keys))

    pointer_voxel_index = torch.searchsorted(voxel_keys, keys)
    maximum = strength.new_zeros(len(voxel_keys))
    maximum.scatter_reduce_(0, pointer_voxel_index, strength, reduce="amax",
                            include_self=True)
    index = torch.arange(len(xyz), device=xyz.device)
    winner = torch.where(strength == maximum[pointer_voxel_index], index,
                         torch.full_like(index, len(xyz)))
    first = torch.full((len(voxel_keys),), len(xyz), device=xyz.device,
                       dtype=torch.long)
    first.scatter_reduce_(0, pointer_voxel_index, winner, reduce="amin",
                          include_self=True)
    return PointerVoxelField(decode_keys(voxel_keys, GRID.shape_zyx), confidence,
                             source_count, source[first], xyz, batch, strength,
                             pointer_voxel_index)

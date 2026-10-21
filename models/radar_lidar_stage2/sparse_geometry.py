"""Fixed Stage-II grid and radar-selected sparse reconstruction coordinates.

Coordinate sets are constructed from Stage I only. Clean LiDAR is used to
populate supervision on those sets, never to select the sets themselves.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from models.radar_lidar_stage1.config import VoxelGrid
from models.radar_lidar_stage1.sparse import encode_keys
from .candidate_domain import CandidateDomain
from .voxel_target import make_targets


@dataclass(frozen=True)
class Stage2Grid:
    minimum_xyz: tuple[float, float, float] = (0.0, -16.0, -2.0)
    maximum_xyz: tuple[float, float, float] = (32.0, 16.0, 2.0)
    voxel_xyz: tuple[float, float, float] = (0.2, 0.2, 0.25)

    @property
    def shape_zyx(self) -> tuple[int, int, int]:
        return (16, 160, 160)

    @property
    def downsampling_zyx(self) -> tuple[tuple[int, int, int], ...]:
        # R2LDM's sparse encoder: 2, 2, (2,1,1), (2,1,1).
        return ((1, 1, 1), (2, 2, 2), (4, 4, 4),
                (8, 4, 4), (16, 4, 4))

    def shape_at(self, level: int) -> tuple[int, int, int]:
        return tuple(dim // stride for dim, stride in
                     zip(self.shape_zyx, self.downsampling_zyx[level]))


GRID = Stage2Grid()


def sorted_unique(coords: torch.Tensor, shape_zyx: tuple[int, int, int]) -> torch.Tensor:
    if coords.ndim != 2 or coords.shape[1] != 4:
        raise ValueError("Expected [N,4] batch,z,y,x sparse coordinates")
    if not len(coords):
        return coords.long()
    keys = torch.unique(encode_keys(coords.long(), shape_zyx), sorted=True)
    z, y, x = shape_zyx
    xx = keys % x
    rest = keys // x
    yy = rest % y
    rest = rest // y
    zz = rest % z
    batch = rest // z
    return torch.stack((batch, zz, yy, xx), dim=1)


def select_allowed(stage1_coords: torch.Tensor, probabilities: torch.Tensor,
                   threshold: float, *, grid: Stage2Grid = GRID,
                   stage1_minimum_xyz: tuple[float, float, float] =
                   (-51.2, -51.2, -2.0),
                   stage1_voxel_xyz: tuple[float, float, float] =
                   (0.2, 0.2, 0.25)) -> tuple[torch.Tensor, torch.Tensor]:
    """Map Stage-I candidates into the exactly aligned Stage-II overlap ROI."""
    if not 0 < threshold < 1:
        raise ValueError("Gate threshold must be in (0,1)")
    if stage1_coords.ndim != 2 or stage1_coords.shape[1] != 4:
        raise ValueError("Stage-I candidate coordinates must be [N,4]")
    scores = probabilities.reshape(-1)
    if len(stage1_coords) != len(scores):
        raise ValueError("Candidate coordinates and probabilities differ")
    if any(abs(a - b) > 1e-6 for a, b in zip(stage1_voxel_xyz, grid.voxel_xyz)):
        raise ValueError("Stage-I and Stage-II voxel sizes must match")
    displacement = [(a - b) / step for a, b, step in
                    zip(grid.minimum_xyz, stage1_minimum_xyz, grid.voxel_xyz)]
    if any(abs(value - round(value)) > 1e-5 for value in displacement):
        raise ValueError("Stage-II origin must align to Stage-I cells")
    shift = stage1_coords.new_tensor(tuple(int(round(v)) for v in displacement[::-1]))
    local = stage1_coords.long().clone()
    local[:, 1:] -= shift
    limits = local.new_tensor(grid.shape_zyx)
    inside = ((local[:, 1:] >= 0) & (local[:, 1:] < limits)).all(-1)
    chosen = inside & (scores >= threshold)
    local, scores = local[chosen], scores[chosen]
    if len(local):
        order = torch.argsort(encode_keys(local, grid.shape_zyx), stable=True)
        local, scores = local[order], scores[order]
        keys = encode_keys(local, grid.shape_zyx)
        if bool((keys[1:] == keys[:-1]).any()):
            raise ValueError("Stage-I map contains duplicate candidate cells")
    return local, scores


def active_hierarchy(fine: torch.Tensor, *, grid: Stage2Grid = GRID
                     ) -> tuple[torch.Tensor, ...]:
    """Allowed sites at fine and R2LDM encoder/decoder scales."""
    if len(fine) and not torch.equal(fine, sorted_unique(fine, grid.shape_zyx)):
        raise ValueError("Fine active sites must be sorted and unique")
    levels = []
    for level, stride in enumerate(grid.downsampling_zyx):
        coords = fine.long().clone()
        coords[:, 1:] //= coords.new_tensor(stride)
        levels.append(sorted_unique(coords, grid.shape_at(level)))
    return tuple(levels)


def gather_features(source_coords: torch.Tensor, source_features: torch.Tensor,
                    query_coords: torch.Tensor, shape_zyx: tuple[int, int, int]
                    ) -> tuple[torch.Tensor, torch.Tensor]:
    """Differentiable coordinate join, with zeros at absent source sites."""
    if source_features.ndim != 2 or len(source_features) != len(source_coords):
        raise ValueError("Sparse features and coordinates differ")
    output = source_features.new_zeros((len(query_coords), source_features.shape[1]))
    found = torch.zeros(len(query_coords), dtype=torch.bool, device=query_coords.device)
    if not len(source_coords) or not len(query_coords):
        return output, found
    source_keys = encode_keys(source_coords.long(), shape_zyx)
    order = torch.argsort(source_keys)
    source_keys = source_keys[order]
    if bool((source_keys[1:] == source_keys[:-1]).any()):
        raise ValueError("Source sparse coordinates contain duplicates")
    query_keys = encode_keys(query_coords.long(), shape_zyx)
    positions = torch.searchsorted(source_keys, query_keys)
    safe = positions.clamp(max=len(source_keys) - 1)
    found = (positions < len(source_keys)) & (source_keys[safe] == query_keys)
    output[found] = source_features[order[safe[found]]]
    return output, found


def parent_features(parent_coords: torch.Tensor, parent_features_: torch.Tensor,
                    child_coords: torch.Tensor, parent_shape: tuple[int, int, int],
                    ratio_zyx: tuple[int, int, int]) -> torch.Tensor:
    parent_query = child_coords.long().clone()
    parent_query[:, 1:] //= parent_query.new_tensor(ratio_zyx)
    result, found = gather_features(parent_coords, parent_features_,
                                    parent_query, parent_shape)
    if not bool(found.all()):
        raise ValueError("Active child has no active parent")
    return result


def voxelize_means(points_xyz: torch.Tensor, *, grid: Stage2Grid = GRID
                   ) -> tuple[torch.Tensor, torch.Tensor]:
    """One XYZ mean per occupied fine cell, sorted in BZYX order."""
    xyz = points_xyz[:, :3]
    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError("LiDAR XYZ must be [N,3]")
    minimum = xyz.new_tensor(grid.minimum_xyz)
    maximum = xyz.new_tensor(grid.maximum_xyz)
    size = xyz.new_tensor(grid.voxel_xyz)
    xyz = xyz[((xyz >= minimum) & (xyz < maximum)).all(-1)]
    if not len(xyz):
        return torch.empty((0, 4), dtype=torch.long, device=points_xyz.device), \
            points_xyz.new_empty((0, 3))
    ijk = torch.floor((xyz - minimum) / size).long()
    coords = torch.cat((torch.zeros((len(ijk), 1), dtype=torch.long,
                                    device=ijk.device), ijk[:, [2, 1, 0]]), dim=1)
    keys = encode_keys(coords, grid.shape_zyx)
    unique, inverse = torch.unique(keys, sorted=True, return_inverse=True)
    sums = xyz.new_zeros((len(unique), 3))
    sums.index_add_(0, inverse, xyz)
    counts = torch.bincount(inverse, minlength=len(unique)).to(xyz.dtype)
    z, y, x = grid.shape_zyx
    xx = unique % x
    rest = unique // x
    yy = rest % y
    zz = (rest // y) % z
    return torch.stack((torch.zeros_like(zz), zz, yy, xx), dim=1), sums / counts[:, None]


def reconstruction_targets(clean_xyz: torch.Tensor, allowed: torch.Tensor,
                           *, grid: Stage2Grid = GRID
                           ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """Measured occupancy, observed free space, and positive offsets on A.

    Missing LiDAR returns are unknown unless a measured first-return ray
    establishes free space. This uses the existing Stage-I visibility rule.
    """
    domain_grid = VoxelGrid(grid.minimum_xyz, grid.maximum_xyz, grid.voxel_xyz)
    domain = CandidateDomain(allowed, clean_xyz.new_zeros(len(allowed)),
                             domain_grid, {})
    clean = clean_xyz[None]
    valid = torch.ones((1, len(clean_xyz)), dtype=torch.bool,
                       device=clean_xyz.device)
    target = make_targets(domain, clean, valid)
    return (target.occupied, target.known_free,
            target.offsets_normalized * clean_xyz.new_tensor(grid.voxel_xyz),
            target.clean_point_coverage)


def pool_allowed_probabilities(fine: torch.Tensor, probabilities: torch.Tensor,
                               coarse: torch.Tensor, stride_zyx: tuple[int, int, int],
                               coarse_shape: tuple[int, int, int]) -> torch.Tensor:
    query = fine.long().clone()
    query[:, 1:] //= query.new_tensor(stride_zyx)
    keys = encode_keys(coarse, coarse_shape)
    positions = torch.searchsorted(keys, encode_keys(query, coarse_shape))
    if len(query) and not bool((positions < len(keys)).all()):
        raise ValueError("Fine active site has no coarse parent")
    pooled = probabilities.new_zeros(len(coarse))
    if len(fine):
        pooled.scatter_reduce_(0, positions, probabilities.reshape(-1),
                               reduce="amax", include_self=True)
    return pooled[:, None]

"""Radar-independent, spatially contiguous full-region LPCR training patches."""

from __future__ import annotations

import hashlib

import torch

from .sparse_geometry import GRID, Stage2Grid, active_hierarchy, voxelize_means
from .sparse_residual import LATENT_CHANNELS, Stage2Conditions


def _keys_to_coords(keys: torch.Tensor,
                    shape_zyx: tuple[int, int, int]) -> torch.Tensor:
    z_size, y_size, x_size = shape_zyx
    x = keys % x_size
    y = keys // x_size % y_size
    z = keys // (x_size * y_size)
    if bool((z >= z_size).any()):
        raise ValueError("LPCR site key exceeds the full reconstruction grid")
    return torch.stack((torch.zeros_like(z), z, y, x), dim=1)


def full_region_training_conditions(
        clean_xyz: torch.Tensor, frame_id: str, epoch: int,
        sites_per_frame: int, *, grid: Stage2Grid = GRID,
        seed: int = 3407) -> Stage2Conditions:
    """Cycle full-grid XY patches across epochs, with a context halo.

    Every fine voxel is eligible regardless of radar. Loss uses all occupied
    core sites and an equal-sized sample of empty core sites. The three-cell
    halo supplies sparse-convolution neighbors.
    """
    if sites_per_frame < 16 * 22 * 22 or epoch < 1:
        raise ValueError("Full-region LPCR needs at least 7,744 sites and epoch >= 1")
    occupied_coords, means = voxelize_means(clean_xyz, grid=grid)
    z_size, y_size, x_size = grid.shape_zyx
    core_side, halo = 16, 3
    if y_size % core_side or x_size % core_side or z_size != 16:
        raise ValueError("LPCR full-region patch requires the configured ROI")
    tiles_y, tiles_x = y_size // core_side, x_size // core_side
    positive_keys = ((occupied_coords[:, 1] * y_size + occupied_coords[:, 2])
                     * x_size + occupied_coords[:, 3])
    digest = hashlib.sha256(f"{seed}:{frame_id}".encode()).digest()
    frame_seed = int.from_bytes(digest[:8], "little") % (2**63)
    generator = torch.Generator(device=clean_xyz.device)
    generator.manual_seed(frame_seed)
    order = torch.randperm(tiles_x * tiles_y, generator=generator,
                           device=clean_xyz.device)
    tile = int(order[(epoch - 1) % len(order)])
    y0, x0 = (tile // tiles_x) * core_side, (tile % tiles_x) * core_side
    y1, x1 = y0 + core_side, x0 + core_side
    yy0, yy1 = max(0, y0 - halo), min(y_size, y1 + halo)
    xx0, xx1 = max(0, x0 - halo), min(x_size, x1 + halo)
    z, y, x = torch.meshgrid(
        torch.arange(z_size, device=clean_xyz.device),
        torch.arange(yy0, yy1, device=clean_xyz.device),
        torch.arange(xx0, xx1, device=clean_xyz.device), indexing="ij")
    fine = torch.stack((torch.zeros_like(z), z, y, x), dim=-1).reshape(-1, 4)
    keys = (fine[:, 1] * y_size + fine[:, 2]) * x_size + fine[:, 3]
    core = ((fine[:, 2] >= y0) & (fine[:, 2] < y1)
            & (fine[:, 3] >= x0) & (fine[:, 3] < x1))
    positions = torch.searchsorted(positive_keys, keys)
    safe = positions.clamp(max=max(len(positive_keys) - 1, 0))
    present = ((positions < len(positive_keys)) & (positive_keys[safe] == keys)
               if len(positive_keys) else torch.zeros_like(core))
    positive = core & present
    offsets = clean_xyz.new_zeros((len(fine), 3))
    centers = (clean_xyz.new_tensor(grid.minimum_xyz)
               + (fine[positive][:, [3, 2, 1]].to(clean_xyz.dtype) + 0.5)
               * clean_xyz.new_tensor(grid.voxel_xyz))
    if bool(positive.any()):
        offsets[positive] = means[positions[positive]] - centers
    negative_indices = torch.where(core & ~positive)[0]
    generator.manual_seed((frame_seed + epoch) % (2**63))
    take = min(len(negative_indices), max(int(positive.sum()), 32))
    chosen_negative = negative_indices[torch.randperm(
        len(negative_indices), generator=generator,
        device=clean_xyz.device)[:take]]
    known_free = torch.zeros_like(core)
    known_free[chosen_negative] = True
    hierarchy = active_hierarchy(fine, grid=grid)
    probabilities = clean_xyz.new_ones((len(fine), 1))
    latent = hierarchy[-1]
    conditions = Stage2Conditions(
        hierarchy, probabilities, clean_xyz.new_ones((len(latent), 1)),
        clean_xyz.new_zeros((len(latent), LATENT_CHANNELS)),
        (positive, known_free, offsets, 1.0))
    return conditions


def full_region_latent_coordinates(*, grid: Stage2Grid = GRID,
                                   device: torch.device | str = "cpu") -> torch.Tensor:
    """All latent sites in the full Stage-II ROI, independent of the gate."""
    shape = grid.shape_at(4)
    count = shape[0] * shape[1] * shape[2]
    return _keys_to_coords(torch.arange(count, device=device), shape)

"""Fixed radar-only Stage-II candidate rule and deterministic radar features."""

from __future__ import annotations

import numpy as np
import torch

from models.radar_lidar_crossmodal_v1.encoder import GRID as RADAR_GRID
from models.radar_lidar_crossmodal_v1.radar_pruner import (
    radar_support_mask, supported_coordinates,
)
from models.radar_lidar_stage1.sparse import encode_keys

from .addition import _voxel_coords
from .sparse_batch import voxelize_means_batched
from .sparse_geometry import GRID, voxelize_means


VERSION = "stage2_geometry_3radarvox_1p6m_v1"
RADIUS_M = 1.6


def geometry_gate_output(radar: torch.Tensor, faulty_xyz: torch.Tensor,
                         support_mask: np.ndarray | None = None) -> dict:
    """All supported cells except measured faulty cells; no learned Stage I.

    Radar features are simple pooled measurements on the old stride-8 grid.
    The trainable Stage-II radar adapter maps these fixed features to its
    latent channels. No clean LiDAR enters the gate or its conditioning.
    """
    if radar.ndim != 2 or radar.shape[1] != 7 or faulty_xyz.ndim != 2 or faulty_xyz.shape[1] != 3:
        raise ValueError("Geometry gate requires radar [N,7] and faulty LiDAR XYZ")
    if radar.device != faulty_xyz.device:
        raise ValueError("Radar and faulty LiDAR must share a device")
    if support_mask is None:
        support_mask = radar_support_mask(radar, RADIUS_M)
    fine = supported_coordinates(support_mask, radar.device)
    fine = torch.cat((torch.zeros((len(fine), 1), device=radar.device,
                                  dtype=torch.long), fine), dim=1)
    observed = _voxel_coords(faulty_xyz, None, GRID)
    if len(fine) and len(observed):
        keep = ~torch.isin(encode_keys(fine, GRID.shape_zyx),
                           encode_keys(observed, GRID.shape_zyx))
        fine = fine[keep]
    shift = torch.tensor([
        round((a - b) / size) for a, b, size in zip(
            GRID.minimum_xyz[::-1], RADAR_GRID.minimum_xyz[::-1],
            GRID.voxel_xyz[::-1])], device=radar.device, dtype=torch.long)
    global_fine = fine.clone()
    global_fine[:, 1:] += shift
    scores = radar.new_ones((len(fine), 1))

    # Deterministic radar statistics: count, RCS, Doppler, age, and the
    # mean position relative to the 8-cell feature bin. Unused columns are
    # zero, preserving the existing trainable 64-to-32 radar adapter.
    minimum = radar.new_tensor(RADAR_GRID.minimum_xyz)
    maximum = radar.new_tensor(RADAR_GRID.maximum_xyz)
    size = radar.new_tensor(RADAR_GRID.size_xyz)
    inside = ((radar[:, :3] >= minimum) & (radar[:, :3] < maximum)).all(1)
    points = radar[inside]
    if len(points):
        fine_index = torch.floor((points[:, :3] - minimum) / size).long()
        coarse_index = fine_index // 8
        unique, inverse = torch.unique(coarse_index[:, [2, 1, 0]], dim=0,
                                       return_inverse=True, sorted=True)
        counts = points.new_zeros(len(unique)).index_add_(
            0, inverse, points.new_ones(len(points)))
        center = minimum + (coarse_index.to(points.dtype) * 8 + 4) * size
        values = torch.stack((
            (points[:, 3] / 30).tanh(),
            (points[:, 4] / 15).tanh(),
            (points[:, 5] / 15).tanh(),
            (points[:, 6] / 20).tanh(),
            *((points[:, :3] - center)[:, axis] / (8 * size[axis])
              for axis in range(3))), dim=1)
        sums = points.new_zeros((len(unique), 7)).index_add_(0, inverse, values)
        features = points.new_zeros((len(unique), 64))
        features[:, 0] = counts.log1p()
        features[:, 1:8] = sums / counts[:, None]
        source_coords = torch.cat((torch.zeros((len(unique), 1),
                                              device=radar.device, dtype=torch.long),
                                   unique), dim=1)
    else:
        source_coords = torch.empty((0, 4), device=radar.device, dtype=torch.long)
        features = radar.new_empty((0, 64))
    return {"confidence_coords": global_fine,
            "confidence_probabilities": scores,
            "conditioning_coords": source_coords,
            "conditioning_features": features}


@torch.no_grad()
def offset_targets(clean_xyz: torch.Tensor, fine: torch.Tensor,
                   clean_batch_ids: torch.Tensor | None = None
                   ) -> tuple[torch.Tensor, torch.Tensor, float]:
    """Measured clean centroids and eligibility, without free-space labels."""
    clean_coords, clean_means = (
        voxelize_means(clean_xyz, grid=GRID) if clean_batch_ids is None else
        voxelize_means_batched(clean_xyz, clean_batch_ids, grid=GRID))
    occupied = torch.zeros(len(fine), dtype=torch.bool, device=fine.device)
    offsets = clean_xyz.new_zeros((len(fine), 3))
    if len(fine) and len(clean_coords):
        clean_keys = encode_keys(clean_coords, GRID.shape_zyx)
        fine_keys = encode_keys(fine, GRID.shape_zyx)
        positions = torch.searchsorted(clean_keys, fine_keys)
        occupied = ((positions < len(clean_keys)) &
                    (clean_keys[positions.clamp(max=len(clean_keys) - 1)] == fine_keys))
        centers = (clean_xyz.new_tensor(GRID.minimum_xyz)
                   + (fine[:, [3, 2, 1]].to(clean_xyz.dtype) + 0.5)
                   * clean_xyz.new_tensor(GRID.voxel_xyz))
        offsets[occupied] = clean_means[positions[occupied]] - centers[occupied]
    coverage = float(occupied.sum()) / max(len(clean_coords), 1)
    return occupied, offsets, coverage

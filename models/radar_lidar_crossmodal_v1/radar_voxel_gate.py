"""Distinct-radar-voxel support on the radar encoder's full spatial grid."""

from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree
import torch

from models.radar_lidar_stage2.candidate_domain import voxel_centers_xyz

from .encoder import GRID
from .radar_pruner import unique_radar_voxel_centers


SUPPORT_POLICY = "radar12vox_1p6m_full_encoder_grid_v1"
MIN_RADAR_VOXELS = 12
RADIUS_M = 1.6


class DistinctRadarVoxelGate:
    """Evaluate a cell using the centers of unique radar voxels in its frame."""

    def __init__(self, radar: torch.Tensor, radar_valid: torch.Tensor,
                 *, radius_m: float = RADIUS_M,
                 min_radar_voxels: int = MIN_RADAR_VOXELS):
        if radar.ndim != 3 or radar.shape[-1] != 7 or radar_valid.shape != radar.shape[:2]:
            raise ValueError("Expected batched seven-column radar points and validity")
        if radius_m <= 0 or min_radar_voxels < 1:
            raise ValueError("Radar gate radius and distinct-voxel count must be positive")
        self.radius_m = radius_m
        self.min_radar_voxels = min_radar_voxels
        self.trees = {}
        for batch in range(len(radar)):
            xyz = radar[batch, radar_valid[batch], :3]
            lower = xyz.new_tensor(GRID.minimum_xyz)
            upper = xyz.new_tensor(GRID.maximum_xyz)
            xyz = xyz[((xyz >= lower) & (xyz < upper)).all(-1)]
            selected = unique_radar_voxel_centers(
                xyz)
            self.trees[batch] = (cKDTree(selected) if len(selected) >= min_radar_voxels
                                 else None)

    def contains_cells(self, coords: torch.Tensor) -> torch.Tensor:
        if coords.ndim != 2 or coords.shape[1] != 4:
            raise ValueError("Expected [N, batch/z/y/x] voxel coordinates")
        source_device = coords.device
        coords = coords.detach().cpu()
        result = np.zeros(len(coords), dtype=np.bool_)
        if not len(coords):
            return torch.from_numpy(result).to(source_device)
        xyz = voxel_centers_xyz(coords, GRID, dtype=torch.float64).numpy()
        batches = coords[:, 0].numpy()
        for batch in np.unique(batches):
            tree = self.trees.get(int(batch))
            if tree is None:
                continue
            selected = np.flatnonzero(batches == batch)
            distance, _ = tree.query(
                xyz[selected], k=self.min_radar_voxels,
                distance_upper_bound=self.radius_m + 1e-9, workers=1)
            kth = distance[:, -1] if distance.ndim == 2 else distance
            result[selected] = kth <= self.radius_m + 1e-9
        return torch.from_numpy(result).to(source_device)

    def count_clean_points(self, clean: torch.Tensor,
                           clean_valid: torch.Tensor) -> int:
        """Count clean returns whose *voxel centers* satisfy this gate."""
        if clean.ndim != 3 or clean.shape[-1] != 4 or clean_valid.shape != clean.shape[:2]:
            raise ValueError("Expected batched four-column LiDAR points and validity")
        xyz = clean[..., :3]
        lower = xyz.new_tensor(GRID.minimum_xyz)
        upper = xyz.new_tensor(GRID.maximum_xyz)
        valid = clean_valid & ((xyz >= lower) & (xyz < upper)).all(-1)
        batch_index, point_index = valid.nonzero(as_tuple=True)
        if not len(batch_index):
            return 0
        indices_xyz = torch.floor((xyz[batch_index, point_index] - lower) /
                                  xyz.new_tensor(GRID.size_xyz)).long()
        coords = torch.cat((batch_index[:, None], indices_xyz[:, [2, 1, 0]]), dim=1)
        return int(self.contains_cells(coords).sum())

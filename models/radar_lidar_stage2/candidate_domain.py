"""Fixed sparse coordinates used to supervise the radar-only confidence head.

This standalone copy retains only the geometry contract needed by
``voxel_target.py``. Legacy Stage-I proposal and interpolation generators
are omitted: the student generates its coordinates from radar.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from models.radar_lidar_stage1.config import VoxelGrid


def voxel_centers_xyz(coordinates: torch.Tensor, grid: VoxelGrid, *,
                      dtype: torch.dtype = torch.float32) -> torch.Tensor:
    xyz = coordinates[:, [3, 2, 1]].to(torch.float64)
    origin = torch.tensor(grid.minimum_xyz, dtype=torch.float64, device=coordinates.device)
    size = torch.tensor(grid.size_xyz, dtype=torch.float64, device=coordinates.device)
    return (origin + (xyz + 0.5) * size).to(dtype)


@dataclass(frozen=True)
class CandidateDomain:
    coordinates: torch.Tensor
    confidence: torch.Tensor
    grid: VoxelGrid
    counts: dict[str, int | float | str]

    @property
    def centers_xyz(self) -> torch.Tensor:
        return voxel_centers_xyz(self.coordinates, self.grid)

    def centers_xyz_numpy(self) -> np.ndarray:
        return self.centers_xyz.detach().cpu().numpy()

    def batch_numpy(self) -> np.ndarray:
        return self.coordinates[:, 0].detach().cpu().numpy()

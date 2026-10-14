"""Online point-cloud augmentation for aligned Stage-I radar/LiDAR pairs."""

from __future__ import annotations

from hashlib import blake2b

import torch

from models.shared_geometric_augmentation import (
    GeometricAugmentationConfig, GeometricTransform, sample_transform,
    transform_points_xy,
)

from .config import VoxelGrid


def frame_generator(seed: int, epoch: int, split: str, frame_id: str) -> torch.Generator:
    """Make augmentation reproducible across workers, shuffle and resumed runs."""
    identity = f"{int(seed)}:{int(epoch)}:{split}/{frame_id}"
    number = int.from_bytes(blake2b(identity.encode(), digest_size=8).digest(), "little")
    return torch.Generator().manual_seed(number & ((1 << 63) - 1))


def augment_pair(radar: torch.Tensor, lidar: torch.Tensor, transform: GeometricTransform,
                 grid: VoxelGrid) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply one XY transform to both sensors, keeping each point's extra channels."""
    if transform.is_identity:
        return radar, lidar
    minimum = radar.new_tensor(grid.minimum_xyz)
    maximum = radar.new_tensor(grid.maximum_xyz)

    def apply(points: torch.Tensor) -> torch.Tensor:
        transformed = transform_points_xy(points, transform)
        inside = ((transformed[:, :3] >= minimum) & (transformed[:, :3] < maximum)).all(1)
        return transformed[inside]

    return apply(radar), apply(lidar)


__all__ = ["GeometricAugmentationConfig", "GeometricTransform", "augment_pair",
           "frame_generator", "sample_transform"]

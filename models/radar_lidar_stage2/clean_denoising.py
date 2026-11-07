"""Clean LiDAR latent denoising inside a fixed radar support mask."""

from __future__ import annotations

import numpy as np
import torch

from models.radar_lidar_crossmodal_v1.radar_pruner import SHAPE

from .sparse_geometry import GRID, active_hierarchy


def clean_points_in_support(clean_xyz: torch.Tensor,
                            mask: np.ndarray) -> torch.Tensor:
    """Keep only clean returns whose 0.2 m/0.25 m cell passes the radar rule."""
    if clean_xyz.ndim != 2 or clean_xyz.shape[1] != 3:
        raise ValueError("Clean LiDAR must be [N,3] XYZ")
    if mask.shape != SHAPE or mask.dtype != np.bool_:
        raise ValueError(f"Support mask must be bool {SHAPE}")
    if clean_xyz.device.type != "cpu":
        raise ValueError("Filter points in the data-loading worker on CPU")
    minimum = clean_xyz.new_tensor(GRID.minimum_xyz)
    maximum = clean_xyz.new_tensor(GRID.maximum_xyz)
    voxel = clean_xyz.new_tensor(GRID.voxel_xyz)
    inside = ((clean_xyz >= minimum) & (clean_xyz < maximum)).all(1)
    xyz = clean_xyz[inside]
    if not len(xyz):
        return xyz
    ijk = torch.floor((xyz - minimum) / voxel).long()
    supported = torch.from_numpy(mask[ijk[:, 2], ijk[:, 1], ijk[:, 0]])
    return xyz[supported]


def batched_hierarchy(fine_zyx: list[torch.Tensor],
                      device: torch.device | str) -> tuple[torch.Tensor, ...]:
    """Combine independent radar-selected coordinate sets without mixing frames."""
    parts = []
    for frame, local in enumerate(fine_zyx):
        if local.ndim != 2 or local.shape[1] != 3:
            raise ValueError("Each support coordinate set must be [N,3] ZYX")
        batch = torch.full((len(local), 1), frame, dtype=torch.long)
        parts.append(torch.cat((batch, local.long()), 1))
    fine = torch.cat(parts, 0).to(device)
    return active_hierarchy(fine)


def frame_mean(values: torch.Tensor, batch_ids: torch.Tensor,
               batch_size: int) -> torch.Tensor:
    if values.ndim != 1 or batch_ids.shape != values.shape:
        raise ValueError("One value and batch ID are required per latent site")
    sums = values.new_zeros(batch_size).scatter_add_(0, batch_ids, values)
    counts = values.new_zeros(batch_size).scatter_add_(
        0, batch_ids, torch.ones_like(values))
    present = counts > 0
    if not bool(present.any()):
        return values.sum() * 0
    return (sums[present] / counts[present]).mean()


def occupancy_counts(logits: torch.Tensor, target: torch.Tensor,
                     threshold: float = 0.5) -> dict[str, int]:
    if logits.shape != target.shape or target.dtype != torch.bool:
        raise ValueError("Occupancy logits and Boolean target must align")
    predicted = logits.sigmoid() >= threshold
    return {"tp": int((predicted & target).sum()),
            "predicted": int(predicted.sum()), "target": int(target.sum())}


def occupancy_metrics(counts: dict[str, int]) -> dict[str, float | int]:
    tp, predicted, target = (counts[key] for key in
                             ("tp", "predicted", "target"))
    return {**counts, "precision": tp / predicted if predicted else 0.0,
            "recall": tp / target if target else 0.0,
            "iou": tp / (predicted + target - tp)
            if predicted + target - tp else 0.0}

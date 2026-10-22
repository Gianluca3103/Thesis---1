"""Full-ROI LPCR inference and R2LDM-style geometric point-cloud metrics.

The decoder queries every fine voxel without radar, Stage I, or clean labels
selecting its candidate coordinates. XY tiles bound sparse-convolution memory.
"""

from __future__ import annotations

import numpy as np
import torch
from scipy.spatial import cKDTree

from .full_lpcr import full_region_latent_coordinates
from .sparse_geometry import GRID, Stage2Grid, active_hierarchy, gather_features


def metric_voxel_centroids(points: np.ndarray, *, grid: Stage2Grid = GRID,
                           metric_voxel_m: float = 0.125) -> np.ndarray:
    """Match the official evaluator's 0.125-m centroid preprocessing in our ROI."""
    xyz = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    lower = np.asarray(grid.minimum_xyz, dtype=np.float32)
    upper = np.asarray(grid.maximum_xyz, dtype=np.float32)
    xyz = xyz[((xyz >= lower) & (xyz < upper)).all(axis=1)]
    if not len(xyz):
        return np.empty((0, 3), dtype=np.float32)
    indices = np.floor((xyz - lower) / metric_voxel_m).astype(np.int32)
    _, inverse = np.unique(indices, axis=0, return_inverse=True)
    sums = np.zeros((int(inverse.max()) + 1, 3), dtype=np.float64)
    counts = np.bincount(inverse)
    np.add.at(sums, inverse, xyz)
    return (sums / counts[:, None]).astype(np.float32)


def point_cloud_metrics(predicted: np.ndarray, target: np.ndarray,
                        *, fscore_radius_m: float = 0.05) -> dict[str, float | int | None]:
    """Symmetric squared-L2 CD/HD and 5-cm F-score used by R2LDM's evaluator."""
    predicted = np.asarray(predicted, dtype=np.float32).reshape(-1, 3)
    target = np.asarray(target, dtype=np.float32).reshape(-1, 3)
    result: dict[str, float | int | None] = {
        "predicted_points": len(predicted), "target_points": len(target),
        "chamfer_squared_l2": None, "hausdorff_squared_l2": None,
        "precision": None, "recall": None, "fscore_5cm": None,
    }
    if not len(predicted) or not len(target):
        return result
    pred_to_gt = cKDTree(target).query(predicted, workers=1)[0]
    gt_to_pred = cKDTree(predicted).query(target, workers=1)[0]
    precision = float(np.mean(pred_to_gt <= fscore_radius_m))
    recall = float(np.mean(gt_to_pred <= fscore_radius_m))
    result.update({
        "chamfer_squared_l2": float(np.mean(pred_to_gt ** 2) + np.mean(gt_to_pred ** 2)),
        "hausdorff_squared_l2": float(max(np.max(pred_to_gt ** 2), np.max(gt_to_pred ** 2))),
        "precision": precision, "recall": recall,
        "fscore_5cm": 2 * precision * recall / (precision + recall + 1e-8),
    })
    return result


def _patch_coordinates(y0: int, y1: int, x0: int, x1: int,
                       *, grid: Stage2Grid, device: torch.device) -> torch.Tensor:
    z_size = grid.shape_zyx[0]
    z, y, x = torch.meshgrid(
        torch.arange(z_size, device=device),
        torch.arange(y0, y1, device=device),
        torch.arange(x0, x1, device=device), indexing="ij")
    zeros = torch.zeros_like(z)
    return torch.stack((zeros, z, y, x), dim=-1).reshape(-1, 4)


@torch.no_grad()
def reconstruct_full_region(model, clean_xyz: torch.Tensor, *,
                            threshold: float = 0.5, tile_size: int = 32,
                            halo: int = 8, grid: Stage2Grid = GRID,
                            return_scores: bool = False):
    """Decode all 409,600 fine cells, returning predicted centroids and count."""
    if not 0 < threshold < 1 or tile_size < 1 or halo < 0:
        raise ValueError("Invalid LPCR inference threshold or tile geometry")
    device = clean_xyz.device
    all_latent_coords = full_region_latent_coordinates(grid=grid, device=device)
    latent, _ = model.lidar_encoder(clean_xyz, all_latent_coords)
    z_size, y_size, x_size = grid.shape_zyx
    predicted = []
    scores = []
    visited = 0
    for y0 in range(0, y_size, tile_size):
        y1 = min(y0 + tile_size, y_size)
        for x0 in range(0, x_size, tile_size):
            x1 = min(x0 + tile_size, x_size)
            fine = _patch_coordinates(max(0, y0 - halo), min(y_size, y1 + halo),
                                      max(0, x0 - halo), min(x_size, x1 + halo),
                                      grid=grid, device=device)
            hierarchy = active_hierarchy(fine, grid=grid)
            tile_latent, found = gather_features(
                all_latent_coords, latent, hierarchy[-1], grid.shape_at(4))
            if not bool(found.all()):
                raise RuntimeError("Full LPCR latent grid is incomplete")
            logits, offsets = model.lpcr(
                tile_latent, hierarchy, clean_xyz.new_ones((len(fine), 1)))
            core = ((fine[:, 2] >= y0) & (fine[:, 2] < y1)
                    & (fine[:, 3] >= x0) & (fine[:, 3] < x1))
            visited += int(core.sum())
            emitted = model.lpcr.emit_points(fine[core], logits[core],
                                             offsets[core], threshold)
            predicted.append(emitted["points_xyz"].cpu())
            if return_scores:
                scores.append(emitted["probabilities"].cpu())
    if visited != z_size * y_size * x_size:
        raise RuntimeError("LPCR full-region tiles did not cover the ROI exactly")
    points = torch.cat(predicted, dim=0)
    return (points, visited, torch.cat(scores, dim=0)) if return_scores else (points, visited)


@torch.no_grad()
def evaluate_full_region(model, clean_xyz: torch.Tensor, *,
                         threshold: float = 0.5, tile_size: int = 32,
                         halo: int = 8) -> dict[str, float | int | None]:
    predicted, visited = reconstruct_full_region(
        model, clean_xyz, threshold=threshold, tile_size=tile_size, halo=halo)
    target = metric_voxel_centroids(clean_xyz.cpu().numpy())
    pred = metric_voxel_centroids(predicted.numpy())
    scores = point_cloud_metrics(pred, target)
    return {"candidate_cells": visited, **scores}

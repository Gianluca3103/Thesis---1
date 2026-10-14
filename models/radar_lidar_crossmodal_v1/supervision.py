"""Oriented VoD object labels and conservative point-dominant voxel identity."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch

from models.radar_lidar_stage1.sparse import SparseSites

from .encoder import GRID


def inverse_augmented_xyz(xyz: np.ndarray, transform: dict | None) -> np.ndarray:
    if not transform:
        return xyz
    yaw = math.radians(transform["yaw_degrees"])
    flip = -1.0 if transform["flip_y"] else 1.0
    scale = transform["scale"]
    matrix = scale * np.array([[math.cos(yaw), -math.sin(yaw) * flip],
                               [math.sin(yaw), math.cos(yaw) * flip]])
    result = xyz.copy()
    offset = np.array([transform["translation_x_m"], transform["translation_y_m"]])
    result[:, :2] = (result[:, :2] - offset) @ np.linalg.inv(matrix).T
    return result


def point_object_ids(xyz_lidar: np.ndarray, label_path: str | Path,
                     camera_from_lidar: np.ndarray) -> tuple[np.ndarray, dict[int, str]]:
    """-1 is unannotated; -2 means overlapping/ambiguous boxes."""
    if camera_from_lidar.shape != (4, 4):
        raise ValueError("Expected homogeneous LiDAR-to-camera calibration")
    camera = np.c_[xyz_lidar, np.ones(len(xyz_lidar))] @ camera_from_lidar.T
    xyz = camera[:, :3]
    matches = np.zeros(len(xyz), dtype=np.int32)
    identity = np.full(len(xyz), -1, dtype=np.int64)
    classes = {}
    for index, line in enumerate(Path(label_path).read_text(encoding="utf-8").splitlines()):
        parts = line.split()
        if not parts:
            continue
        if len(parts) < 15:
            raise ValueError(f"Malformed KITTI box: {line}")
        cls = parts[0]
        if cls == "DontCare":
            continue
        height, width, length = map(float, parts[8:11])
        x, y, z = map(float, parts[11:14])
        yaw = float(parts[14])
        dx, dy, dz = xyz[:, 0] - x, xyz[:, 1] - y, xyz[:, 2] - z
        local_x = math.cos(yaw) * dx - math.sin(yaw) * dz
        local_z = math.sin(yaw) * dx + math.cos(yaw) * dz
        inside = ((abs(local_x) <= width / 2) & (abs(local_z) <= length / 2)
                  & (dy >= -height) & (dy <= 0))
        matches += inside
        identity[inside] = index
        classes[index] = cls
    identity[matches > 1] = -2
    return identity, classes


def voxel_object_ids(sites: SparseSites, points: torch.Tensor, valid: torch.Tensor,
                     label_paths: list[Path], camera_matrices: list[np.ndarray],
                     transforms: list[dict | None] | None = None, *,
                     min_dominance: float = 0.8, max_radar_age: float | None = None):
    """Use underlying points, avoiding center-only labels on large bottleneck voxels."""
    if len(label_paths) != len(points) or len(camera_matrices) != len(points):
        raise ValueError("Annotation metadata and batch size differ")
    transforms = transforms or [None] * len(points)
    if len(transforms) != len(points) or not 0.5 <= min_dominance <= 1:
        raise ValueError("Invalid transforms or dominance threshold")
    result = np.full(len(sites.coords), -1, dtype=np.int64)
    classes = {}
    coordinates = sites.coords.detach().cpu().numpy()
    lookup = {tuple(row): index for index, row in enumerate(coordinates)}
    minimum = np.asarray(GRID.minimum_xyz)
    step = np.asarray(GRID.size_xyz) * sites.stride
    for batch in range(len(points)):
        selected = points[batch, valid[batch]].detach().cpu().numpy()
        if not len(selected):
            continue
        xyz = selected[:, :3]
        point_ids, names = point_object_ids(inverse_augmented_xyz(xyz, transforms[batch]),
                                            label_paths[batch], camera_matrices[batch])
        if max_radar_age is not None:
            if selected.shape[1] < 7:
                raise ValueError("Radar age filter requires the 7-column stacked radar input")
            point_ids[selected[:, 6] > max_radar_age] = -1
        voxel_xyz = np.floor((xyz - minimum) / step).astype(np.int64)
        grouped = {}
        for local_index, (x, y, z) in enumerate(voxel_xyz):
            site = lookup.get((batch, z, y, x))
            if site is not None:
                grouped.setdefault(site, []).append(int(point_ids[local_index]))
        for site, ids in grouped.items():
            candidates, counts = np.unique(ids, return_counts=True)
            positive = candidates >= 0
            if not positive.any():
                continue
            selected_idx = np.argmax(np.where(positive, counts, -1))
            if counts[selected_idx] / len(ids) < min_dominance or -2 in ids:
                continue
            source_id = int(candidates[selected_idx])
            global_id = batch * 1_000_000 + source_id
            result[site] = global_id
            classes[global_id] = names[source_id]
    return torch.as_tensor(result, device=sites.coords.device), classes

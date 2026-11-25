"""Clean View-of-Delft LiDAR occupancy on the existing Stage-II voxel grid."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
from torch.utils.data import Dataset

from Fault_Localization_Model.vod_dataset.vod_io import (
    discover_vod_frames, load_vod_lidar, load_vod_radar_to_lidar,
)
from models.radar_lidar_stage1.radar_fov import crop_lidar_to_radar_fov
from models.radar_lidar_stage2.sparse_geometry import GRID


def voxelize_clean_lidar(points: np.ndarray, lidar_from_radar: np.ndarray) -> np.ndarray:
    """Return unique XYZ voxel indices; never cap the number of LiDAR points."""
    cropped = crop_lidar_to_radar_fov(points, lidar_from_radar)
    xyz = np.asarray(cropped[:, :3], dtype=np.float32)
    minimum = np.asarray(GRID.minimum_xyz, dtype=np.float32)
    maximum = np.asarray(GRID.maximum_xyz, dtype=np.float32)
    size = np.asarray(GRID.voxel_xyz, dtype=np.float32)
    inside = np.all((xyz >= minimum) & (xyz < maximum), axis=1)
    cells = np.floor((xyz[inside] - minimum) / size).astype(np.int32)
    shape_xyz = np.asarray(GRID.shape_zyx[::-1], dtype=np.int32)
    cells = cells[np.all((cells >= 0) & (cells < shape_xyz), axis=1)]
    return np.unique(cells, axis=0).astype(np.int32, copy=False)


def occupancy_counts(predicted: np.ndarray, target: np.ndarray) -> dict[str, int]:
    """Count exact occupied-voxel matches for one frame, using XYZ indices."""
    shape = np.asarray(GRID.shape_zyx[::-1])
    for name, values in (("predicted", predicted), ("target", target)):
        if values.ndim != 2 or values.shape[1] != 3:
            raise ValueError(f"{name} coordinates must have shape [N,3]")
        if len(values) and (np.any(values < 0) or np.any(values >= shape)):
            raise ValueError(f"{name} contains coordinates outside the Stage-II grid")
    dimensions = tuple(int(value) for value in shape)
    p = np.unique(np.ravel_multi_index(predicted.T, dimensions)) if len(predicted) else np.empty(0, np.int64)
    t = np.unique(np.ravel_multi_index(target.T, dimensions)) if len(target) else np.empty(0, np.int64)
    return {"tp": int(np.intersect1d(p, t, assume_unique=True).size),
            "predicted": int(len(p)), "target": int(len(t))}


def summarize_counts(counts: dict[str, int]) -> dict[str, float | int]:
    tp, predicted, target = (int(counts[key]) for key in ("tp", "predicted", "target"))
    union = predicted + target - tp
    return {**counts, "exact_iou": tp / union if union else 1.0,
            "exact_f1": 2 * tp / (predicted + target) if predicted + target else 1.0,
            "precision": tp / predicted if predicted else 0.0,
            "recall": tp / target if target else 0.0}


class CleanVoxelFrames(Dataset):
    """Official VoD split; radar calibration is used only for the fixed FOV crop."""

    def __init__(self, vod_root: str | Path, split: str, *, cache_root: str | Path,
                 limit: int | None = None):
        all_frames = discover_vod_frames(vod_root, split, radar_variant="radar")
        self.frames = all_frames
        if limit is not None:
            if limit < 1:
                raise ValueError("Frame limit must be positive")
            self.frames = self.frames[:limit]
        self.root = Path(cache_root) / split
        self.root.mkdir(parents=True, exist_ok=True)
        manifest = self.root / "manifest.json"
        expected = {"version": 1, "grid": {"minimum_xyz": GRID.minimum_xyz,
                    "maximum_xyz": GRID.maximum_xyz, "voxel_xyz": GRID.voxel_xyz},
                    "crop": "calibrated_radar_fov_70deg_80m", "split": split,
                    "frame_ids": [frame.frame_id for frame in all_frames]}
        if manifest.exists():
            if json.loads(manifest.read_text(encoding="utf-8")) != json.loads(json.dumps(expected)):
                raise ValueError(f"Incompatible voxel cache manifest: {manifest}")
        else:
            temporary = manifest.with_name(manifest.name + f".{os.getpid()}.tmp")
            temporary.write_text(json.dumps(expected, indent=2), encoding="utf-8")
            temporary.replace(manifest)

    def __len__(self) -> int:
        return len(self.frames)

    def __getitem__(self, index: int) -> dict:
        frame = self.frames[index]
        path = self.root / f"{frame.frame_id}.npz"
        if path.is_file():
            with np.load(path, allow_pickle=False) as saved:
                cells = saved["cells"]
        else:
            transform = load_vod_radar_to_lidar(
                frame.lidar_calibration_path, frame.radar_calibration_path)
            cells = voxelize_clean_lidar(load_vod_lidar(frame.lidar_path), transform)
            temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
            try:
                with temporary.open("wb") as stream:
                    np.savez_compressed(stream, cells=cells)
                temporary.replace(path)
            finally:
                temporary.unlink(missing_ok=True)
        if cells.dtype != np.int32 or cells.ndim != 2 or cells.shape[1] != 3 or not len(cells):
            raise ValueError(f"Empty or corrupt voxel cache for {frame.frame_id}")
        if np.any(cells < 0) or np.any(cells >= np.asarray(GRID.shape_zyx[::-1])):
            raise ValueError(f"Out-of-bounds voxel cache for {frame.frame_id}")
        return {"frame_id": frame.frame_id, "cells": cells}

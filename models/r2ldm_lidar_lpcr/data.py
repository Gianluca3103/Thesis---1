"""VoD LiDAR and deterministic XYZ voxel means for the independent LPCR run.

The geometry matches R2LDM's released ``VoxelFeatureConfig``. No radar,
cross-attention features, occupancy labels, or existing model cache enter here.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from Fault_Localization_Model.vod_dataset.vod_io import (
    load_vod_lidar, resolve_vod_public_root,
)


@dataclass(frozen=True)
class R2LDMGrid:
    minimum_xyz: tuple[float, float, float] = (0.0, -16.0, -2.0)
    maximum_xyz: tuple[float, float, float] = (32.0, 16.0, 4.0)
    voxel_xyz: tuple[float, float, float] = (0.1, 0.1, 0.15)
    max_points_in_voxel: int = 8
    max_voxels: int = 40000

    @property
    def shape_zyx(self) -> tuple[int, int, int]:
        extent = (np.asarray(self.maximum_xyz) - np.asarray(self.minimum_xyz))
        shape_xyz = np.rint(extent / np.asarray(self.voxel_xyz)).astype(np.int64)
        return tuple(int(value) for value in shape_xyz[::-1])

    def validate(self) -> None:
        values = np.asarray((*self.minimum_xyz, *self.maximum_xyz, *self.voxel_xyz))
        if not np.isfinite(values).all() or any(size <= 0 for size in self.voxel_xyz):
            raise ValueError("LPCR grid must contain finite positive voxel sizes")
        if any(low >= high for low, high in zip(self.minimum_xyz, self.maximum_xyz)):
            raise ValueError("LPCR grid extents must be positive")
        cells = ((np.asarray(self.maximum_xyz) - np.asarray(self.minimum_xyz)) /
                 np.asarray(self.voxel_xyz))
        if not np.allclose(cells, np.rint(cells), rtol=0, atol=1e-5):
            raise ValueError("LPCR voxel sizes must divide all grid extents")
        if self.shape_zyx != (40, 320, 320):
            raise ValueError("The released LPCR encoder and decoder require a 40x320x320 grid")
        if self.max_points_in_voxel < 1 or self.max_voxels < 1:
            raise ValueError("LPCR voxel limits must be positive")


def voxelize_xyz(points_xyz: np.ndarray, grid: R2LDMGrid) -> tuple[np.ndarray, np.ndarray]:
    """Return sorted unique ZYX cells and mean XYZ of up to eight points/cell.

    The original R2LDM reader averages the points retained by its voxelizer.
    The first 40,000 encountered cells and first eight points within each cell
    are kept. This NumPy implementation is deterministic, but not bitwise
    identical to the upstream compiled voxelizer.
    """
    grid.validate()
    xyz = np.asarray(points_xyz, dtype=np.float32)
    if xyz.ndim != 2 or xyz.shape[1] != 3 or not np.isfinite(xyz).all():
        raise ValueError("LiDAR XYZ must be a finite [N,3] array")
    minimum = np.asarray(grid.minimum_xyz, dtype=np.float32)
    maximum = np.asarray(grid.maximum_xyz, dtype=np.float32)
    step = np.asarray(grid.voxel_xyz, dtype=np.float32)
    inside = np.all((xyz >= minimum) & (xyz < maximum), axis=1)
    xyz = xyz[inside]
    if not len(xyz):
        return np.empty((0, 3), np.int32), np.empty((0, 3), np.float32)
    ijk = np.floor((xyz - minimum) / step).astype(np.int32)
    depth, height, width = grid.shape_zyx
    bounded = np.all((ijk >= 0) & (ijk < np.asarray((width, height, depth))), axis=1)
    xyz, ijk = xyz[bounded], ijk[bounded]
    keys = (ijk[:, 2].astype(np.int64) * height + ijk[:, 1]) * width + ijk[:, 0]
    all_unique, first_positions = np.unique(keys, return_index=True)
    kept = np.sort(all_unique[np.argsort(first_positions, kind="stable")[:grid.max_voxels]])
    order = np.argsort(keys, kind="stable")
    sorted_keys = keys[order]
    _, starts, groups = np.unique(sorted_keys, return_index=True, return_inverse=True)
    rank = np.arange(len(sorted_keys)) - starts[groups]
    selected = np.isin(sorted_keys, kept, assume_unique=False)
    selected &= rank < grid.max_points_in_voxel
    kept_groups = np.searchsorted(kept, sorted_keys[selected])
    sums = np.zeros((len(kept), 3), dtype=np.float64)
    np.add.at(sums, kept_groups, xyz[order[selected]])
    counts = np.bincount(kept_groups, minlength=len(kept))
    means = (sums / counts[:, None]).astype(np.float32)
    z = kept // (height * width)
    y = (kept // width) % height
    x = kept % width
    coords = np.stack((z, y, x), axis=1).astype(np.int32)
    return coords, means


class VoDLidarLPCRDataset(Dataset):
    """Official train/val split, LiDAR files only, in LiDAR coordinates."""

    def __init__(self, vod_root: str | Path, split: str, *,
                 grid: R2LDMGrid = R2LDMGrid(), limit: int | None = None,
                 cache_root: str | Path | None = None):
        if split not in ("train", "val"):
            raise ValueError("LPCR dataset supports official train and val splits")
        grid.validate()
        public = resolve_vod_public_root(vod_root)
        split_file = public / "lidar" / "ImageSets" / f"{split}.txt"
        frame_ids = [line.strip() for line in split_file.read_text(encoding="utf-8").splitlines()
                     if line.strip()]
        all_frame_ids = frame_ids.copy()
        if limit is not None:
            if limit < 1:
                raise ValueError("LPCR split limit must be positive")
            frame_ids = frame_ids[:limit]
        if not all_frame_ids or len(all_frame_ids) != len(set(all_frame_ids)):
            raise ValueError(f"Empty or duplicate frame IDs in {split_file}")
        training = public / "lidar" / "training" / "velodyne"
        testing = public / "lidar" / "testing" / "velodyne"
        self.frames = [(frame_id, next((path for path in
                        (training / f"{frame_id}.bin", testing / f"{frame_id}.bin")
                        if path.is_file()), None)) for frame_id in frame_ids]
        missing = [frame_id for frame_id, path in self.frames if path is None]
        if missing:
            raise FileNotFoundError(f"Missing LiDAR frames in {split}: {missing[:5]}")
        self.grid = grid
        self.split = split
        self.cache_dir = None if cache_root is None else Path(cache_root) / split
        if self.cache_dir is not None:
            serializable_grid = json.loads(json.dumps(asdict(grid)))
            expected = {"version": 1, "vod_root": str(public.resolve()),
                        "grid": serializable_grid, "frame_ids": all_frame_ids,
                        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
            manifest = self.cache_dir / "manifest.json"
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            if manifest.exists():
                if json.loads(manifest.read_text(encoding="utf-8")) != expected:
                    raise ValueError(f"LPCR voxel cache differs from this data or code: {manifest}")
            else:
                temporary = manifest.with_name(manifest.name + f".{os.getpid()}.tmp")
                temporary.write_text(json.dumps(expected, indent=2), encoding="utf-8")
                temporary.replace(manifest)

    def __len__(self) -> int:
        return len(self.frames)

    def check_complete(self) -> None:
        if self.cache_dir is None:
            raise ValueError("LPCR dataset has no voxel cache root")
        missing = [frame_id for frame_id, _ in self.frames
                   if not (self.cache_dir / f"{frame_id}.npz").is_file()]
        if missing:
            raise FileNotFoundError(f"LPCR {self.split} cache lacks {len(missing)} frames; "
                                    f"first {missing[0]}")

    def __getitem__(self, index: int) -> dict:
        frame_id, path = self.frames[index]
        saved = None if self.cache_dir is None else self.cache_dir / f"{frame_id}.npz"
        if saved is not None and saved.is_file():
            with np.load(saved, allow_pickle=False) as data:
                coords, means = data["coords"], data["means"]
        else:
            coords, means = voxelize_xyz(load_vod_lidar(path)[:, :3], self.grid)
            if saved is not None:
                temporary = saved.with_name(saved.name + f".{os.getpid()}.tmp")
                try:
                    with temporary.open("wb") as stream:
                        np.savez_compressed(stream, coords=coords, means=means)
                    temporary.replace(saved)
                finally:
                    temporary.unlink(missing_ok=True)
        if (coords.ndim != 2 or coords.shape[1] != 3 or coords.dtype != np.int32
                or means.shape != coords.shape or means.dtype != np.float32
                or not np.isfinite(means).all()):
            raise ValueError(f"Corrupt LPCR voxel cache for frame {frame_id}")
        if not len(coords):
            raise ValueError(f"LiDAR frame {frame_id} has no points in the LPCR grid")
        return {"frame_id": frame_id, "coords": torch.from_numpy(coords),
                "means": torch.from_numpy(means)}


def collate_lpcr(rows: list[dict]) -> dict:
    if not rows:
        raise ValueError("LPCR batch cannot be empty")
    coords = []
    for batch_index, row in enumerate(rows):
        batch = torch.full((len(row["coords"]), 1), batch_index, dtype=torch.int32)
        coords.append(torch.cat((batch, row["coords"]), dim=1))
    return {"frame_id": [row["frame_id"] for row in rows],
            "coords": torch.cat(coords, dim=0),
            "means": torch.cat([row["means"] for row in rows], dim=0),
            "batch_size": len(rows)}

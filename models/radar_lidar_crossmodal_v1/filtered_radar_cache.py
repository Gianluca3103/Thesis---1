"""Exact post-filter radar returns for radar-to-radar reconstruction.

The filter is evaluated once on the original frame. No LiDAR data or labels
enter this cache. Its manifest binds it to the existing fixed radar frames.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import torch

from .encoder import GRID
from .radar_voxel_gate import (MIN_RADAR_VOXELS, RADIUS_M,
                               SUPPORT_POLICY, filter_radar_valid)


VERSION = "filtered_radar_self_reconstruction_v1"


def expected_manifest(raw_root: Path, split: str) -> dict:
    path = Path(raw_root) / split / "manifest.json"
    source = json.loads(path.read_text(encoding="utf-8"))
    if source.get("split") != split or not source.get("frame_ids"):
        raise ValueError(f"Invalid fixed-frame manifest: {path}")
    return {
        "version": VERSION,
        "split": split,
        "raw_manifest_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "frame_ids": source["frame_ids"],
        "filter_policy": SUPPORT_POLICY,
        "min_distinct_radar_voxels": MIN_RADAR_VOXELS,
        "radius_m": RADIUS_M,
        "grid_minimum_xyz": list(GRID.minimum_xyz),
        "grid_maximum_xyz": list(GRID.maximum_xyz),
        "grid_size_xyz": list(GRID.size_xyz),
    }


def filtered_frame(raw_root: Path, split: str, frame_id: str) -> dict:
    saved = torch.load(Path(raw_root) / split / "radar" / f"{frame_id}.pt",
                       map_location="cpu", weights_only=True)
    points = saved.get("points")
    if (saved.get("frame_id") != frame_id or not isinstance(points, torch.Tensor)
            or points.ndim != 2 or points.shape[1] != 7
            or not bool(torch.isfinite(points).all())):
        raise ValueError(f"Invalid raw radar frame: {split}/{frame_id}")
    valid = torch.ones((1, len(points)), dtype=torch.bool)
    kept = filter_radar_valid(points[None], valid)[0]
    return {"version": VERSION, "frame_id": frame_id,
            "points": points[kept].contiguous(), "raw_returns": len(points)}


class FilteredRadarCache:
    def __init__(self, root: Path, raw_root: Path, split: str, *, create: bool = False):
        self.root = Path(root) / split
        self.expected = expected_manifest(raw_root, split)
        self.frame_ids = self.expected["frame_ids"]
        path = self.root / "manifest.json"
        if create:
            self.root.mkdir(parents=True, exist_ok=True)
            if not path.exists():
                temporary = path.with_suffix(".json.tmp")
                temporary.write_text(json.dumps(self.expected), encoding="utf-8")
                temporary.replace(path)
        if not path.is_file() or json.loads(path.read_text(encoding="utf-8")) != self.expected:
            raise ValueError(f"Filtered radar cache manifest differs: {path}")

    def path(self, frame_id: str) -> Path:
        if frame_id not in self.frame_ids:
            raise ValueError(f"Unknown frame ID: {frame_id}")
        return self.root / f"{frame_id}.pt"

    def write(self, row: dict) -> bool:
        path = self.path(row["frame_id"])
        if path.is_file():
            return False
        temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        try:
            torch.save(row, temporary)
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)
        return True

    def check_complete(self) -> None:
        missing = [frame_id for frame_id in self.frame_ids if not self.path(frame_id).is_file()]
        if missing:
            raise FileNotFoundError(f"Missing {len(missing)} filtered radar frames; first: {missing[0]}")

    def __len__(self) -> int:
        return len(self.frame_ids)

    def __getitem__(self, index: int) -> dict:
        frame_id = self.frame_ids[index]
        row = torch.load(self.path(frame_id), map_location="cpu", weights_only=True)
        points = row.get("points")
        if (row.get("version") != VERSION or row.get("frame_id") != frame_id
                or not isinstance(points, torch.Tensor) or points.ndim != 2
                or points.shape[1] != 7):
            raise ValueError(f"Invalid filtered radar cache frame: {frame_id}")
        return row

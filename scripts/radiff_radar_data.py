"""Five-sweep VoD radar data for a 4D-RaDiff-style VAE experiment.

The paper specifies the five-sweep input, its spatial crop, and the five
features. The paper does not publish its exact feature-normalization bounds
or fixed point count; those choices are recorded by the training script.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch

from Fault_Localization_Model.vod_dataset.vod_io import (
    load_vod_radar, load_vod_split_ids, resolve_vod_public_root,
    vod_partition_for_split,
)


VERSION = "radiff_v2_vod5_xyz_compensated_doppler_rcs_v1"
RADAR_VARIANT = "radar_5frames"
FEATURES = ("x", "y", "z", "compensated_doppler", "rcs")
RANGE_MIN = np.array((0.0, -25.6, -3.0), dtype=np.float32)
RANGE_MAX = np.array((51.2, 25.6, 2.0), dtype=np.float32)


def select_paper_radar(radar: np.ndarray) -> np.ndarray:
    """Crop native VoD radar coordinates and select the paper's five fields."""
    radar = np.asarray(radar, dtype=np.float32)
    if radar.ndim != 2 or radar.shape[1] != 7 or not np.isfinite(radar).all():
        raise ValueError("Expected finite VoD radar [N,7]")
    inside = ((radar[:, :3] >= RANGE_MIN) &
              (radar[:, :3] < RANGE_MAX)).all(axis=1)
    # VoD fields: XYZ, RCS, raw radial velocity, compensated radial velocity,
    # scan age. RaDiff uses XYZ, compensated Doppler, RCS in that order.
    return np.ascontiguousarray(radar[inside][:, (0, 1, 2, 5, 3)])


def expected_manifest(vod_root: Path, split: str) -> dict:
    public = resolve_vod_public_root(vod_root)
    ids = load_vod_split_ids(public, split)
    partition = vod_partition_for_split(public, split, ids)
    return {"version": VERSION, "split": split, "frame_ids": ids,
            "radar_variant": RADAR_VARIANT, "partition": partition,
            "features": list(FEATURES), "range_min_xyz": RANGE_MIN.tolist(),
            "range_max_xyz": RANGE_MAX.tolist(),
            "source_split_sha256": hashlib.sha256(
                (public / "lidar" / "ImageSets" / f"{split}.txt").read_bytes()).hexdigest()}


class RadarFiveFrames:
    def __init__(self, cache_root: Path, split: str, limit: int | None = None):
        self.root = Path(cache_root) / split
        self.manifest_path = self.root / "manifest.json"
        self.manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        if (self.manifest.get("version") != VERSION or
                self.manifest.get("split") != split or
                self.manifest.get("radar_variant") != RADAR_VARIANT or
                tuple(self.manifest.get("features", ())) != FEATURES or
                tuple(self.manifest.get("range_min_xyz", ())) != tuple(RANGE_MIN) or
                tuple(self.manifest.get("range_max_xyz", ())) != tuple(RANGE_MAX)):
            raise ValueError(f"Wrong 4D-RaDiff radar cache: {self.manifest_path}")
        self.ids = self.manifest["frame_ids"][:limit]
        missing = [frame_id for frame_id in self.ids
                   if not (self.root / f"{frame_id}.pt").is_file()]
        if missing:
            raise FileNotFoundError(f"{len(missing)} radar cache frames missing; first {missing[0]}")

    def __len__(self) -> int:
        return len(self.ids)

    def load(self, index: int) -> torch.Tensor:
        frame_id = self.ids[index]
        row = torch.load(self.root / f"{frame_id}.pt", map_location="cpu",
                         weights_only=True)
        points = row.get("points")
        if (row.get("version") != VERSION or row.get("frame_id") != frame_id or
                not isinstance(points, torch.Tensor) or points.ndim != 2 or
                points.shape[1] != 5 or not bool(torch.isfinite(points).all())):
            raise ValueError(f"Invalid 4D-RaDiff radar frame: {frame_id}")
        return points.float()

    def sha256(self) -> str:
        return hashlib.sha256(self.manifest_path.read_bytes()).hexdigest()


def cache_split(vod_root: Path, cache_root: Path, split: str,
                limit: int | None = None) -> dict:
    public = resolve_vod_public_root(vod_root)
    manifest = expected_manifest(public, split)
    if limit is not None:
        manifest["frame_ids"] = manifest["frame_ids"][:limit]
        manifest["limit"] = limit
    target = Path(cache_root) / split
    target.mkdir(parents=True, exist_ok=True)
    manifest_path = target / "manifest.json"
    if manifest_path.is_file():
        if json.loads(manifest_path.read_text(encoding="utf-8")) != manifest:
            raise ValueError(f"Radar cache manifest differs: {manifest_path}")
    else:
        temp = manifest_path.with_suffix(".json.tmp")
        temp.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        temp.replace(manifest_path)
    written = empty = retained = raw_count = 0
    for frame_id in manifest["frame_ids"]:
        path = target / f"{frame_id}.pt"
        if path.is_file():
            continue
        source = (public / RADAR_VARIANT / manifest["partition"] /
                  "velodyne" / f"{frame_id}.bin")
        radar = load_vod_radar(source)
        points = select_paper_radar(radar)
        temp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        try:
            torch.save({"version": VERSION, "frame_id": frame_id,
                        "points": torch.from_numpy(points),
                        "raw_returns": len(radar)}, temp)
            temp.replace(path)
        finally:
            temp.unlink(missing_ok=True)
        written += 1
        empty += len(points) == 0
        retained += len(points)
        raw_count += len(radar)
    frames = RadarFiveFrames(cache_root, split)
    return {"split": split, "frames": len(frames), "written": written,
            "reused": len(frames) - written, "new_empty": empty,
            "new_retained_returns": retained, "new_raw_returns": raw_count,
            "manifest_sha256": frames.sha256()}


class GlobalScale:
    """Train-only global min/max normalization into [-1, 1]."""

    def __init__(self, minimum: list[float], maximum: list[float]):
        self.minimum = torch.tensor(minimum, dtype=torch.float32)
        self.maximum = torch.tensor(maximum, dtype=torch.float32)
        if (self.minimum.shape != (5,) or self.maximum.shape != (5,) or
                not bool(torch.all(self.maximum > self.minimum))):
            raise ValueError("Invalid five-feature normalization bounds")

    @classmethod
    def fit(cls, frames: RadarFiveFrames) -> "GlobalScale":
        minimum = torch.full((5,), float("inf"))
        maximum = torch.full((5,), float("-inf"))
        for i in range(len(frames)):
            points = frames.load(i)
            if len(points):
                minimum = torch.minimum(minimum, points.amin(0))
                maximum = torch.maximum(maximum, points.amax(0))
        if not bool(torch.isfinite(minimum).all()):
            raise ValueError("All training radar frames are empty")
        maximum = torch.maximum(maximum, minimum + 1e-6)
        return cls(minimum.tolist(), maximum.tolist())

    def encode(self, points: torch.Tensor) -> torch.Tensor:
        low = self.minimum.to(points.device)
        high = self.maximum.to(points.device)
        return ((points - low) * (2 / (high - low)) - 1).clamp(-1, 1)

    def decode(self, points: torch.Tensor) -> torch.Tensor:
        low = self.minimum.to(points.device)
        high = self.maximum.to(points.device)
        return (points + 1) * ((high - low) / 2) + low

    def to_json(self) -> dict:
        return {"minimum": self.minimum.tolist(), "maximum": self.maximum.tolist(),
                "source": "training frames only; published bounds unavailable"}

    @classmethod
    def from_json(cls, data: dict) -> "GlobalScale":
        return cls(data["minimum"], data["maximum"])

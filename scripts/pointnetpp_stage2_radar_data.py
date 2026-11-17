"""Aligned 20-sweep radar, filtered and cropped exactly for the Stage-II ROI."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import torch
from tqdm import tqdm

from models.radar_lidar_stage1.data import VoDStage1Dataset
from models.radar_lidar_stage2.sparse_geometry import GRID as STAGE2_GRID
from models.radar_lidar_crossmodal_v1.radar_voxel_gate import (
    MIN_RADAR_VOXELS, RADIUS_M, SUPPORT_POLICY, filter_radar_valid,
)


VERSION = "pointnetpp_stage2_radar20_gate12_crop_v1"
RADAR_VARIANT = "radar_20frames_verified_doppler_radial"
COLUMNS = ("x", "y", "z", "rcs", "raw_radial", "compensated_radial", "scan_age")


def _source_hash() -> str:
    repo = Path(__file__).resolve().parents[1]
    digest = hashlib.sha256()
    for relative in (
        "scripts/pointnetpp_stage2_radar_data.py",
        "models/radar_lidar_crossmodal_v1/radar_voxel_gate.py",
        "models/radar_lidar_crossmodal_v1/radar_pruner.py",
        "models/radar_lidar_stage2/sparse_geometry.py",
        "models/radar_lidar_stage1/data.py",
        "Fault_Localization_Model/vod_dataset/vod_io.py",
    ):
        digest.update((repo / relative).read_bytes())
    return digest.hexdigest()


def expected_manifest(dataset: VoDStage1Dataset, limit: int | None = None) -> dict:
    frames = dataset.frames[:limit]
    return {
        "version": VERSION,
        "split": frames[0].split,
        "vod_root": str(Path(frames[0].lidar_path).resolve().parents[3]),
        "radar_variant": RADAR_VARIANT,
        "frame_ids": [frame.frame_id for frame in frames],
        "limit": limit,
        "columns": list(COLUMNS),
        "stage2_minimum_xyz": list(STAGE2_GRID.minimum_xyz),
        "stage2_maximum_xyz": list(STAGE2_GRID.maximum_xyz),
        "voxel_xyz": list(STAGE2_GRID.voxel_xyz),
        "filter_policy": SUPPORT_POLICY,
        "min_distinct_radar_voxels": MIN_RADAR_VOXELS,
        "radius_m": RADIUS_M,
        "order": "filter original aligned frame, then crop retained points",
        "source_sha256": _source_hash(),
    }


def filter_and_crop(aligned_radar: torch.Tensor) -> torch.Tensor:
    if (aligned_radar.ndim != 2 or aligned_radar.shape[1] != 7 or
            not torch.isfinite(aligned_radar).all()):
        raise ValueError("Expected finite aligned radar [N,7]")
    valid = torch.ones((1, len(aligned_radar)), dtype=torch.bool)
    kept = filter_radar_valid(aligned_radar[None], valid)[0]
    points = aligned_radar[kept]
    minimum = points.new_tensor(STAGE2_GRID.minimum_xyz)
    maximum = points.new_tensor(STAGE2_GRID.maximum_xyz)
    inside = ((points[:, :3] >= minimum) & (points[:, :3] < maximum)).all(1)
    return points[inside].contiguous()


def cache_split(vod_root: Path, cache_root: Path, split: str,
                limit: int | None = None) -> dict:
    if limit is not None and limit < 1:
        raise ValueError("Limit must be positive")
    dataset = VoDStage1Dataset(vod_root, split, radar_variant=RADAR_VARIANT,
                               include_clean=False)
    expected = expected_manifest(dataset, limit)
    target = cache_root / split
    target.mkdir(parents=True, exist_ok=True)
    manifest_path = target / "manifest.json"
    if manifest_path.exists():
        if json.loads(manifest_path.read_text(encoding="utf-8")) != expected:
            raise ValueError(f"Incompatible radar cache: {manifest_path}")
    else:
        temporary = manifest_path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(expected, indent=2) + "\n", encoding="utf-8")
        temporary.replace(manifest_path)
    written = reused = empty = raw_returns = retained_returns = 0
    for index in tqdm(range(len(expected["frame_ids"])), desc=f"Stage-II radar {split}",
                      unit="frame", dynamic_ncols=True):
        frame_id = expected["frame_ids"][index]
        path = target / f"{frame_id}.pt"
        if path.is_file():
            reused += 1
            continue
        raw = dataset[index]["radar"]
        points = filter_and_crop(raw)
        temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        try:
            torch.save({"version": VERSION, "frame_id": frame_id,
                        "points": points, "raw_returns": len(raw)}, temporary)
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)
        written += 1
        empty += len(points) == 0
        raw_returns += len(raw)
        retained_returns += len(points)
    return {"split": split, "frames": len(expected["frame_ids"]),
            "written": written, "reused": reused, "empty_new": empty,
            "raw_returns_new": raw_returns, "retained_returns_new": retained_returns,
            "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest()}


class Stage2RadarFrames:
    def __init__(self, root: Path, split: str, limit: int | None = None):
        self.root = Path(root) / split
        self.manifest_path = self.root / "manifest.json"
        self.manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        if (self.manifest.get("version") != VERSION or
                self.manifest.get("radar_variant") != RADAR_VARIANT or
                self.manifest.get("stage2_minimum_xyz") != list(STAGE2_GRID.minimum_xyz) or
                self.manifest.get("stage2_maximum_xyz") != list(STAGE2_GRID.maximum_xyz) or
                self.manifest.get("filter_policy") != SUPPORT_POLICY or
                self.manifest.get("min_distinct_radar_voxels") != MIN_RADAR_VOXELS or
                self.manifest.get("radius_m") != RADIUS_M):
            raise ValueError(f"Wrong Stage-II radar cache: {self.manifest_path}")
        self.ids = self.manifest["frame_ids"][:limit]
        missing = [frame_id for frame_id in self.ids
                   if not (self.root / f"{frame_id}.pt").is_file()]
        if missing:
            raise FileNotFoundError(f"Missing {len(missing)} filtered radar frames; first {missing[0]}")

    def __len__(self) -> int:
        return len(self.ids)

    def load(self, index: int) -> torch.Tensor:
        frame_id = self.ids[index]
        saved = torch.load(self.root / f"{frame_id}.pt", map_location="cpu",
                           weights_only=True)
        points = saved.get("points")
        if (saved.get("version") != VERSION or saved.get("frame_id") != frame_id or
                not isinstance(points, torch.Tensor) or points.ndim != 2 or
                points.shape[1] != len(COLUMNS) or not torch.isfinite(points).all()):
            raise ValueError(f"Invalid radar cache frame: {frame_id}")
        minimum = points.new_tensor(STAGE2_GRID.minimum_xyz)
        maximum = points.new_tensor(STAGE2_GRID.maximum_xyz)
        if len(points) and not bool(((points[:, :3] >= minimum) &
                                     (points[:, :3] < maximum)).all()):
            raise ValueError(f"Radar points outside Stage-II ROI: {frame_id}")
        return points.float()

    def sha256(self) -> str:
        return hashlib.sha256(self.manifest_path.read_bytes()).hexdigest()


class Stage2RadarScale:
    def __init__(self, minimum: list[float], maximum: list[float]):
        self.minimum = torch.tensor(minimum, dtype=torch.float32)
        self.maximum = torch.tensor(maximum, dtype=torch.float32)
        if (self.minimum.shape != (7,) or self.maximum.shape != (7,) or
                not bool(torch.all(self.maximum > self.minimum))):
            raise ValueError("Expected valid seven-column scale")

    @classmethod
    def fit(cls, frames: Stage2RadarFrames) -> "Stage2RadarScale":
        low = torch.full((7,), float("inf"))
        high = torch.full((7,), float("-inf"))
        for index in range(len(frames)):
            points = frames.load(index)
            if len(points):
                low = torch.minimum(low, points.amin(0))
                high = torch.maximum(high, points.amax(0))
        if not torch.isfinite(low).all():
            raise ValueError("No filtered radar returns in training cache")
        low[:3] = torch.tensor(STAGE2_GRID.minimum_xyz)
        high[:3] = torch.tensor(STAGE2_GRID.maximum_xyz)
        high = torch.maximum(high, low + 1e-6)
        return cls(low.tolist(), high.tolist())

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
                "source": "training split only; XYZ fixed to Stage-II grid"}

    @classmethod
    def from_json(cls, value: dict) -> "Stage2RadarScale":
        return cls(value["minimum"], value["maximum"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vod-root", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "val", "test"), required=True)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    print(json.dumps(cache_split(args.vod_root, args.cache_root, args.split, args.limit),
                     indent=2), flush=True)


if __name__ == "__main__":
    main()

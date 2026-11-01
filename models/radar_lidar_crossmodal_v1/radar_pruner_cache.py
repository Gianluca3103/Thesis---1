"""Compact, versioned radar-support masks for the fixed 20-frame dataset."""

from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path

import numpy as np

from models.radar_lidar_stage2.sparse_geometry import GRID
from .radar_pruner import (CELL_COUNT, SHAPE, SUPPORT_MIN_VOXELS,
                           SUPPORT_RADIUS_M, VERSION)


def identity(split: str, frame_ids: list[str], radar_variant: str,
             *, data_cache_root: Path | None = None) -> dict:
    result = {"version": VERSION + "_support_cache_v1", "split": split,
            "frame_ids": frame_ids, "radar_variant": radar_variant,
            "grid": asdict(GRID), "radius_m": SUPPORT_RADIUS_M,
            "min_distinct_radar_voxels": SUPPORT_MIN_VOXELS}
    if data_cache_root is not None:
        manifest = Path(data_cache_root) / split / "manifest.json"
        result["fixed_frame_manifest_sha256"] = hashlib.sha256(
            manifest.read_bytes()).hexdigest()
    return result


class RadarSupportCache:
    def __init__(self, root: Path, expected: dict, *, create: bool = False):
        self.root = Path(root)
        manifest = self.root / "manifest.json"
        if create and not manifest.exists():
            self.root.mkdir(parents=True, exist_ok=True)
            manifest.write_text(json.dumps(expected, indent=2), encoding="utf-8")
        if json.loads(manifest.read_text(encoding="utf-8")) != json.loads(
                json.dumps(expected)):
            raise ValueError(f"Radar support cache policy differs: {manifest}")

    def path(self, frame_id: str) -> Path:
        if not frame_id.isdigit():
            raise ValueError("Frame ID must be numeric")
        return self.root / f"{frame_id}.npz"

    def write(self, frame_id: str, mask: np.ndarray) -> None:
        values = np.asarray(mask)
        if values.shape != SHAPE or values.dtype != np.bool_:
            raise ValueError("Invalid radar support mask")
        packed = np.packbits(values.ravel(), bitorder="little")
        target = self.path(frame_id)
        temporary = target.with_name(target.name + f".{os.getpid()}.tmp")
        try:
            with temporary.open("wb") as stream:
                np.savez_compressed(stream, packed=packed,
                                    supported=np.int64(values.sum()))
            temporary.replace(target)
        finally:
            temporary.unlink(missing_ok=True)

    def load(self, frame_id: str) -> np.ndarray:
        with np.load(self.path(frame_id), allow_pickle=False) as archive:
            packed = archive["packed"]
            supported = int(archive["supported"])
        if packed.dtype != np.uint8 or packed.shape != ((CELL_COUNT + 7) // 8,):
            raise ValueError(f"Invalid packed radar support: {frame_id}")
        mask = np.unpackbits(packed, bitorder="little", count=CELL_COUNT)
        if int(mask.sum()) != supported:
            raise ValueError(f"Radar support count mismatch: {frame_id}")
        return mask.reshape(SHAPE).astype(np.bool_)

    def check_complete(self) -> None:
        manifest = json.loads((self.root / "manifest.json").read_text(encoding="utf-8"))
        missing = [item for item in manifest["frame_ids"] if not self.path(item).is_file()]
        if missing:
            raise FileNotFoundError(f"Missing {len(missing)} radar masks; first {missing[0]}")


def bind_faulty_cache_identity(base: dict, faulty_root: Path, split: str) -> dict:
    """Bind a Stage-II cache to the faulty input used by this pruner."""
    manifest = Path(faulty_root) / split / "manifest.json"
    return {**base, "faulty_root": str(Path(faulty_root).resolve()),
            "faulty_manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest()}

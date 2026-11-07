"""Persistent fixed ray associations for the geometry-only Stage-II baseline."""

from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch

from models.radar_lidar_stage1.sparse import encode_keys

from .geometry_gate import VERSION as GATE_VERSION
from .sparse_geometry import GRID


VERSION = "stage2_geometry_ray_indices_v1"


def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def identity(split: str, frame_ids: list[str], *, data_cache_root: Path,
             faulty_root: Path, support_cache_root: Path,
             fault_region_root: Path, radar_variant: str,
             gate_threshold: float, angular_tolerance_deg: float,
             lateral_tolerance_m: float, range_margin_m: float | None,
             device: str) -> dict:
    from . import ray_loss

    return {
        "version": VERSION, "split": split, "frame_ids": frame_ids,
        "radar_variant": radar_variant, "grid": asdict(GRID),
        "geometry_gate_version": GATE_VERSION,
        "gate_threshold": gate_threshold,
        "angular_tolerance_deg": angular_tolerance_deg,
        "lateral_tolerance_m": lateral_tolerance_m,
        "range_margin_m": range_margin_m,
        "label_device": device,
        "hit_enabled": True, "extra_enabled": True,
        "ray_loss_source_sha256": _sha256(Path(ray_loss.__file__)),
        "fixed_manifest_sha256": _sha256(data_cache_root / split / "manifest.json"),
        "faulty_manifest_sha256": _sha256(faulty_root / split / "manifest.json"),
        "support_manifest_sha256": _sha256(
            support_cache_root / split / "manifest.json"),
        "region_manifest_sha256": _sha256(
            fault_region_root / split / "manifest.json"),
        "region_data_sha256": _sha256(
            fault_region_root / split / "regions.json"),
    }


def coordinate_sha256(fine: torch.Tensor) -> str:
    if fine.ndim != 2 or fine.shape[1] != 4:
        raise ValueError("Ray label coordinates must be BZYX cells")
    keys = encode_keys(fine.long(), GRID.shape_zyx).detach().cpu().numpy()
    return hashlib.sha256(np.asarray(keys, dtype="<i8").tobytes()).hexdigest()


class RayLabelCache:
    def __init__(self, root: Path, expected: dict, *, create: bool = False):
        self.root = Path(root)
        manifest = self.root / "manifest.json"
        if create and not manifest.exists():
            self.root.mkdir(parents=True, exist_ok=True)
            manifest.write_text(json.dumps(expected, indent=2), encoding="utf-8")
        if json.loads(manifest.read_text(encoding="utf-8")) != json.loads(
                json.dumps(expected)):
            raise ValueError(f"Ray label cache policy differs: {manifest}")
        self.frame_ids = set(expected["frame_ids"])

    def path(self, frame_id: str) -> Path:
        if frame_id not in self.frame_ids:
            raise ValueError(f"Ray label frame is outside manifest: {frame_id}")
        return self.root / f"{frame_id}.npz"

    def write(self, frame_id: str, fine: torch.Tensor, labels: dict) -> None:
        count = len(fine)
        if int(labels["candidate_count"]) != count:
            raise ValueError("Ray labels differ from candidate coordinates")
        arrays = {}
        for name in ("hit", "front", "extra"):
            values = np.asarray(labels[name])
            if (values.ndim != 1 or values.dtype.kind not in "iu"
                    or bool(np.any(values < 0))
                    or bool(np.any(values >= count))):
                raise ValueError(f"Invalid ray {name} indices")
            arrays[name] = values.astype(np.int32)
        target = self.path(frame_id)
        temporary = target.with_name(target.name + f".{os.getpid()}.tmp")
        try:
            with temporary.open("wb") as stream:
                np.savez_compressed(
                    stream, candidate_count=np.int64(count),
                    coordinate_sha256=np.asarray(coordinate_sha256(fine)),
                    **arrays)
            temporary.replace(target)
        finally:
            temporary.unlink(missing_ok=True)

    def load(self, frame_id: str, fine: torch.Tensor) -> dict:
        with np.load(self.path(frame_id), allow_pickle=False) as archive:
            count = int(archive["candidate_count"])
            digest = str(archive["coordinate_sha256"].item())
            arrays = {name: archive[name].copy()
                      for name in ("hit", "front", "extra")}
        if count != len(fine) or digest != coordinate_sha256(fine):
            raise ValueError(f"Ray label coordinates differ: {frame_id}")
        for name, values in arrays.items():
            if (values.dtype != np.int32 or values.ndim != 1
                    or bool(np.any(values < 0))
                    or bool(np.any(values >= count))):
                raise ValueError(f"Invalid cached ray {name}: {frame_id}")
        return {"candidate_count": count, **arrays}

    def check_complete(self) -> None:
        missing = [frame_id for frame_id in self.frame_ids
                   if not self.path(frame_id).is_file()]
        if missing:
            raise FileNotFoundError(
                f"Ray label cache missing {len(missing)} frames, first {missing[0]}")

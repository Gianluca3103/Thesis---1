"""Lossless compressed fixed Stage-I outputs and clean LiDAR targets."""

from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch

from models.radar_lidar_crossmodal_v1.encoder import GRID as STAGE1_GRID

from .sparse_geometry import GRID, reconstruction_targets, select_allowed
from .sparse_residual import VERSION
from .stage2_io import sha256


CACHE_VERSION = VERSION + "_cache_v1"


def _source_hash() -> str:
    digest = hashlib.sha256()
    for filename in ("sparse_geometry.py", "voxel_target.py",
                     "candidate_domain.py", "stage2_cache.py"):
        digest.update((Path(__file__).parent / filename).read_bytes())
    return digest.hexdigest()


def cache_identity(stage1_checkpoint: Path, vod_root: Path, radar_variant: str,
                   radius_m: float, gate_threshold: float, split: str,
                   frame_ids: list[str]) -> dict:
    return {"version": CACHE_VERSION, "source_sha256": _source_hash(),
            "stage1_sha256": sha256(stage1_checkpoint),
            "vod_root": str(vod_root.resolve()),
            "radar_variant": radar_variant, "radius_m": radius_m,
            "gate_threshold": gate_threshold, "grid": asdict(GRID),
            "split": split, "frame_ids": frame_ids}


class Stage2FixedCache:
    def __init__(self, root: Path, expected: dict, *, create: bool = False):
        self.root = root
        expected = json.loads(json.dumps(expected))
        manifest_path = root / "manifest.json"
        if create and not manifest_path.exists():
            root.mkdir(parents=True, exist_ok=True)
            manifest_path.write_text(json.dumps(expected, indent=2), encoding="utf-8")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest != expected:
            raise ValueError(f"Stage-II cache manifest differs: {manifest_path}")

    def path(self, frame_id: str) -> Path:
        if not frame_id.isdigit():
            raise ValueError("VoD frame ID must be numeric")
        return self.root / f"{frame_id}.npz"

    def check_complete(self) -> None:
        manifest = json.loads((self.root / "manifest.json").read_text(encoding="utf-8"))
        missing = [item for item in manifest["frame_ids"] if not self.path(item).is_file()]
        if missing:
            raise FileNotFoundError(f"Stage-II cache missing {len(missing)} frames, first {missing[0]}")

    def write(self, frame_id: str, output: dict, clean: torch.Tensor,
              threshold: float) -> None:
        fine, scores = select_allowed(output["confidence_coords"],
                                      output["confidence_probabilities"], threshold)
        occupied, known_free, offsets, coverage = reconstruction_targets(clean, fine)
        global_coords = fine.clone()
        shift_xyz = [round((a - b) / step) for a, b, step in
                     zip(GRID.minimum_xyz, STAGE1_GRID.minimum_xyz, GRID.voxel_xyz)]
        global_coords[:, 1:] += global_coords.new_tensor(shift_xyz[::-1])
        selected = {"confidence_coords": global_coords,
                    "confidence_probabilities": scores[:, None],
                    "conditioning_coords": output["conditioning_coords"],
                    "conditioning_features": output["conditioning_features"],
                    "occupied": occupied, "known_free": known_free,
                    "offsets": offsets}
        arrays = {key: value.detach().cpu().numpy() for key, value in selected.items()}
        for key in ("confidence_coords", "conditioning_coords"):
            if arrays[key].size and (arrays[key].min() < -32768 or
                                     arrays[key].max() > 32767):
                raise ValueError("Cached sparse coordinates exceed lossless int16 range")
            arrays[key] = arrays[key].astype(np.int16)
        arrays["coverage"] = np.array(coverage, dtype=np.float64)
        target = self.path(frame_id)
        temporary = target.with_name(target.name + f".{os.getpid()}.tmp")
        try:
            with temporary.open("wb") as stream:
                np.savez_compressed(stream, **arrays)
            temporary.replace(target)
        finally:
            temporary.unlink(missing_ok=True)

    def load(self, frame_id: str, device: str) -> tuple[dict, tuple]:
        with np.load(self.path(frame_id), allow_pickle=False) as saved:
            arrays = {key: torch.from_numpy(saved[key].copy()).to(device)
                      for key in ("confidence_coords", "confidence_probabilities",
                                  "conditioning_coords", "conditioning_features",
                                  "occupied", "known_free", "offsets")}
            coverage = float(saved["coverage"])
        output = {key: arrays[key] for key in
                  ("confidence_coords", "confidence_probabilities",
                   "conditioning_coords", "conditioning_features")}
        output["confidence_coords"] = output["confidence_coords"].long()
        output["conditioning_coords"] = output["conditioning_coords"].long()
        targets = (arrays["occupied"].bool(), arrays["known_free"].bool(),
                   arrays["offsets"], coverage)
        return output, targets

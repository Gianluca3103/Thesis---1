"""Unaugmented, aligned VoD frames shared by all three training phases."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import torch


VERSION = 1


def _source_hash() -> str:
    repo = Path(__file__).resolve().parents[2]
    files = (repo / "models/radar_lidar_stage1/data.py",
             repo / "Fault_Localization_Model/vod_dataset/vod_io.py",
             Path(__file__))
    digest = hashlib.sha256()
    for path in files:
        digest.update(path.read_bytes())
    return digest.hexdigest()


class FixedFrameCache:
    def __init__(self, root: Path, dataset, *, create: bool = False,
                 include_clean: bool = True):
        if dataset.augmentation_config.enabled:
            raise ValueError("Fixed frame cache requires geometric augmentation to be disabled")
        self.root = Path(root)
        self.dataset = dataset
        self.frames = dataset.frames
        self.include_clean = include_clean
        expected = {
            "version": VERSION,
            "vod_root": str(Path(dataset.frames[0].lidar_path).resolve().parents[3]),
            "split": dataset.frames[0].split,
            "radar_variant": dataset.frames[0].radar_variant,
            "frame_ids": [frame.frame_id for frame in dataset.frames],
            "source_hash": _source_hash(),
            "augmentation_enabled": False,
        }
        manifest = self.root / "manifest.json"
        if create:
            self.root.mkdir(parents=True, exist_ok=True)
            if not manifest.exists():
                temporary = manifest.with_suffix(".json.tmp")
                temporary.write_text(json.dumps(expected), encoding="utf-8")
                temporary.replace(manifest)
        if not manifest.is_file() or json.loads(manifest.read_text(encoding="utf-8")) != expected:
            raise ValueError("Fixed frame cache manifest differs from VoD data or source code")

    def path(self, frame_id: str, modality: str) -> Path:
        if not frame_id.isdigit() or modality not in ("radar", "clean_lidar"):
            raise ValueError("Invalid fixed frame cache key")
        return self.root / modality / f"{frame_id}.pt"

    def write(self, row: dict) -> int:
        total = 0
        for modality in ("radar", "clean_lidar"):
            path = self.path(row["frame_id"], modality)
            if path.is_file():
                total += path.stat().st_size
                continue
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
            try:
                torch.save({"version": VERSION, "frame_id": row["frame_id"],
                            "points": row[modality]}, temporary)
                temporary.replace(path)
            finally:
                temporary.unlink(missing_ok=True)
            total += path.stat().st_size
        return total

    def check_complete(self) -> None:
        for frame in self.frames:
            for modality in (("radar", "clean_lidar") if self.include_clean else ("radar",)):
                if not self.path(frame.frame_id, modality).is_file():
                    raise FileNotFoundError(f"Missing fixed frame cache: {frame.frame_id} {modality}")

    def __len__(self) -> int:
        return len(self.frames)

    def set_epoch(self, epoch: int) -> None:
        self.dataset.set_epoch(epoch)

    def __getitem__(self, index: int) -> dict:
        frame = self.frames[index]
        result = {"frame_id": frame.frame_id, "split": frame.split}
        for modality in (("radar", "clean_lidar") if self.include_clean else ("radar",)):
            saved = torch.load(self.path(frame.frame_id, modality), map_location="cpu",
                               weights_only=True)
            points = saved.get("points")
            if (saved.get("version") != VERSION or saved.get("frame_id") != frame.frame_id
                    or not isinstance(points, torch.Tensor) or points.ndim != 2
                    or points.shape[1] != (7 if modality == "radar" else 4)):
                raise ValueError(f"Corrupt fixed frame cache: {frame.frame_id} {modality}")
            result[modality] = points
        return result

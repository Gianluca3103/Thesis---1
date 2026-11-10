"""Fixed FPS/collapsed-neighborhood cache for the radar D-PCC adaptation."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import torch

from .filtered_radar_cache import FilteredRadarCache
from .radar_dpcc import K, RATES, build_hierarchy


VERSION = "radar_dpcc_geometry_v1"


class RadarDPCCCache:
    def __init__(self, filtered_root: Path, raw_root: Path, geometry_root: Path,
                 split: str, *, create=False):
        self.filtered = FilteredRadarCache(filtered_root, raw_root, split)
        self.root = Path(geometry_root) / split
        self.frame_ids = self.filtered.frame_ids
        source = Path(filtered_root) / split / "manifest.json"
        self.manifest = {"version": VERSION, "split": split,
                         "filtered_manifest_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                         "frame_ids": self.frame_ids, "rates": list(RATES), "k": K}
        path = self.root / "manifest.json"
        if create:
            self.root.mkdir(parents=True, exist_ok=True)
            if not path.exists():
                path.write_text(json.dumps(self.manifest), encoding="utf-8")
        if not path.is_file() or json.loads(path.read_text(encoding="utf-8")) != self.manifest:
            raise ValueError(f"D-PCC geometry cache manifest differs: {path}")

    def path(self, frame_id: str):
        return self.root / f"{frame_id}.pt"

    def build(self, index: int):
        row = self.filtered[index]
        return {"version": VERSION, "frame_id": row["frame_id"],
                "hierarchy": build_hierarchy(row["points"])}

    def write(self, row: dict):
        path = self.path(row["frame_id"])
        if path.exists():
            return False
        temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        try:
            torch.save(row, temporary)
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)
        return True

    def check_complete(self):
        missing = [name for name in self.frame_ids if not self.path(name).is_file()]
        if missing:
            raise FileNotFoundError(f"Missing {len(missing)} D-PCC geometries; first {missing[0]}")

    def __len__(self):
        return len(self.frame_ids)

    def __getitem__(self, index):
        row = self.filtered[index]
        geometry = torch.load(self.path(row["frame_id"]), map_location="cpu", weights_only=True)
        if geometry.get("version") != VERSION or geometry.get("frame_id") != row["frame_id"]:
            raise ValueError(f"Invalid D-PCC geometry cache: {row['frame_id']}")
        return {"frame_id": row["frame_id"], "points": row["points"],
                "hierarchy": geometry["hierarchy"]}

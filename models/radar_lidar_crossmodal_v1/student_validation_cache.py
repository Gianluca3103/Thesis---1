"""Reusable, coordinate-checked supervision for fixed train or validation frames."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path

import torch

from models.radar_lidar_stage1.sparse import SparseSites

from .encoder import GRID


VERSION = 2


@dataclass(frozen=True)
class CachedTargets:
    occupied: torch.Tensor
    known_free: torch.Tensor
    clean_points_in_candidates: int
    clean_points_in_grid: int


def source_fingerprint() -> str:
    root = Path(__file__).resolve().parent
    repo = root.parents[1]
    files = (root / "student_validation_cache.py", root / "student_loss.py",
             root / "student.py", root / "teacher.py", root / "encoder.py",
             root / "previous_encoder.py", root / "pipeline_data.py",
             repo / "models/radar_lidar_stage2/voxel_target.py",
             repo / "models/radar_lidar_stage2/candidate_domain.py",
             repo / "models/radar_lidar_stage1/correspondence.py",
             repo / "models/radar_lidar_stage1/sparse.py")
    digest = hashlib.sha256()
    for path in files:
        digest.update(path.read_bytes())
    return digest.hexdigest()


class StudentValidationCache:
    def __init__(self, root: Path, teacher_sha256: str, radar_variant: str,
                 *, split: str = "val", create: bool = False):
        if split not in ("train", "val"):
            raise ValueError("Student cache split must be train or val")
        self.root = Path(root)
        expected = {"version": VERSION, "teacher_sha256": teacher_sha256,
                    "radar_variant": radar_variant, "split": split,
                    "augmentation_enabled": False,
                    "source_fingerprint": source_fingerprint(),
                    "grid_minimum_xyz": list(GRID.minimum_xyz),
                    "grid_size_xyz": list(GRID.size_xyz),
                    "grid_shape_zyx": list(GRID.shape_zyx)}
        manifest = self.root / "manifest.json"
        if create:
            self.root.mkdir(parents=True, exist_ok=True)
            if not manifest.exists():
                temporary = manifest.with_suffix(".json.tmp")
                temporary.write_text(json.dumps(expected, indent=2), encoding="utf-8")
                temporary.replace(manifest)
        if not manifest.is_file() or json.loads(manifest.read_text(encoding="utf-8")) != expected:
            raise ValueError("Student cache manifest differs from split, teacher, grid, or source code")

    def path(self, frame_id: str) -> Path:
        if not frame_id.isdigit():
            raise ValueError("Expected a numeric VoD frame ID")
        return self.root / f"{frame_id}.pt"

    def write(self, frame_id: str, student_output, teacher_output, targets,
              evidence: torch.Tensor) -> int:
        path = self.path(frame_id)
        if path.exists():
            self._read(frame_id)
            return path.stat().st_size
        candidate = student_output.confidence_logits.coords
        bottleneck = student_output.conditioning.coords
        if (candidate[:, 0].ne(0).any() or bottleneck[:, 0].ne(0).any()
                or teacher_output.fused.coords[:, 0].ne(0).any()):
            raise ValueError("Cache entries must be built one frame at a time")
        if not torch.equal(bottleneck, teacher_output.fused.coords):
            raise ValueError("Teacher and student bottleneck coordinates differ")
        entry = {"version": VERSION, "frame_id": frame_id,
                 "candidate_coords": candidate.detach().to("cpu", dtype=torch.int32),
                 "bottleneck_coords": bottleneck.detach().to("cpu", dtype=torch.int32),
                 "teacher_features": teacher_output.fused.features.detach().cpu(),
                 "evidence": evidence.detach().cpu(),
                 "occupied": targets.occupied.detach().cpu(),
                 "known_free": targets.known_free.detach().cpu(),
                 "clean_points_in_candidates": targets.clean_points_in_candidates,
                 "clean_points_in_grid": targets.clean_points_in_grid}
        temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
        try:
            torch.save(entry, temporary)
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)
        return path.stat().st_size

    def _read(self, frame_id: str) -> dict:
        entry = torch.load(self.path(frame_id), map_location="cpu", weights_only=True)
        if entry.get("version") != VERSION or entry.get("frame_id") != frame_id:
            raise ValueError(f"Invalid validation cache entry: {frame_id}")
        candidates = entry["candidate_coords"]
        bottleneck = entry["bottleneck_coords"]
        if (candidates.ndim != 2 or candidates.shape[1] != 4
                or bottleneck.ndim != 2 or bottleneck.shape[1] != 4
                or any(entry[key].shape != (len(candidates),)
                       for key in ("evidence", "occupied", "known_free"))
                or entry["teacher_features"].shape != (len(bottleneck), 64)
                or candidates[:, 0].ne(0).any() or bottleneck[:, 0].ne(0).any()):
            raise ValueError(f"Invalid validation cache shapes: {frame_id}")
        return entry

    def load_batch(self, frame_ids: list[str], student_output, device: str):
        entries = [self._read(frame_id) for frame_id in frame_ids]
        candidate_coords = torch.cat([
            torch.cat((entry["candidate_coords"][:, :1] + batch,
                       entry["candidate_coords"][:, 1:]), dim=1)
            for batch, entry in enumerate(entries)
        ], dim=0).to(device=device, dtype=torch.long)
        bottleneck_coords = torch.cat([
            torch.cat((entry["bottleneck_coords"][:, :1] + batch,
                       entry["bottleneck_coords"][:, 1:]), dim=1)
            for batch, entry in enumerate(entries)
        ], dim=0).to(device=device, dtype=torch.long)
        if (not torch.equal(candidate_coords, student_output.confidence_logits.coords)
                or not torch.equal(bottleneck_coords, student_output.conditioning.coords)):
            raise ValueError("Cached candidate coordinates differ from current radar-only output")
        features = torch.cat([entry["teacher_features"] for entry in entries]).to(device)
        teacher = SparseSites._from_validated(bottleneck_coords, features,
                                              GRID.scale_shape(8), 8)
        targets = CachedTargets(
            occupied=torch.cat([entry["occupied"] for entry in entries]).to(device),
            known_free=torch.cat([entry["known_free"] for entry in entries]).to(device),
            clean_points_in_candidates=sum(entry["clean_points_in_candidates"] for entry in entries),
            clean_points_in_grid=sum(entry["clean_points_in_grid"] for entry in entries))
        evidence = torch.cat([entry["evidence"] for entry in entries]).to(device)
        return teacher, targets, evidence

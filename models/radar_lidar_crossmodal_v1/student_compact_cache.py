"""Lossless, versioned storage for fixed student supervision.

This is separate from the v2 cache so an interrupted v2 build can resume with
its original manifest and files. Only storage changes; the decoded tensors are
identical to those consumed by StudentValidationCache.load_batch.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import zlib

import numpy as np
import torch

from .encoder import GRID
from .student_validation_cache import StudentValidationCache, source_fingerprint


VERSION = 3
DTYPES = {"torch.float16": torch.float16, "torch.bfloat16": torch.bfloat16,
          "torch.float32": torch.float32, "torch.float64": torch.float64}


def _compress(raw: bytes, codec: str, level: int) -> bytes:
    if codec == "zlib":
        return zlib.compress(raw, level)
    if codec == "zstd":
        try:
            import zstandard
        except ImportError as exc:
            raise RuntimeError("zstd cache format requires pip install zstandard") from exc
        return zstandard.ZstdCompressor(level=level, write_checksum=True).compress(raw)
    raise ValueError(f"Unsupported student cache codec: {codec}")


def _decompress(data: bytes, codec: str) -> bytes:
    if codec == "zlib":
        return zlib.decompress(data)
    if codec == "zstd":
        try:
            import zstandard
        except ImportError as exc:
            raise RuntimeError("zstd cache format requires pip install zstandard") from exc
        return zstandard.ZstdDecompressor().decompress(data)
    raise ValueError(f"Unsupported student cache codec: {codec}")


def _spatial_keys(coords: torch.Tensor, shape: tuple[int, int, int], dtype) -> np.ndarray:
    rows = coords.detach().to(device="cpu", dtype=torch.int64).numpy()
    if rows.ndim != 2 or rows.shape[1] != 4 or np.any(rows[:, 0] != 0):
        raise ValueError("Compact cache requires one frame of BZYX coordinates")
    if np.any(rows[:, 1:] < 0) or np.any(rows[:, 1:] >= np.asarray(shape)):
        raise ValueError("Sparse coordinate lies outside its grid")
    z, y, x = rows[:, 1], rows[:, 2], rows[:, 3]
    keys = (z * shape[1] + y) * shape[2] + x
    if len(keys) > 1 and np.any(keys[1:] <= keys[:-1]):
        raise ValueError("Sparse cache coordinates must be sorted and unique")
    if np.prod(shape) - 1 > np.iinfo(dtype).max:
        raise ValueError("Coordinate grid does not fit packed integer type")
    return keys.astype(dtype, copy=False)


def _pack_coords(coords: torch.Tensor, shape: tuple[int, int, int], dtype,
                 codec: str, level: int) -> bytes:
    keys = _spatial_keys(coords, shape, dtype)
    deltas = np.empty_like(keys)
    if len(keys):
        deltas[0] = keys[0]
        deltas[1:] = keys[1:] - keys[:-1]
    return _compress(deltas.tobytes(), codec, level)


def _unpack_coords(blob: bytes, length: int, shape: tuple[int, int, int],
                   dtype, codec: str) -> torch.Tensor:
    raw = _decompress(blob, codec)
    if len(raw) != length * np.dtype(dtype).itemsize:
        raise ValueError("Corrupt compact coordinate payload")
    delta = np.frombuffer(raw, dtype=dtype)
    keys = np.cumsum(delta, dtype=np.uint64)
    if np.any(keys >= np.prod(shape)) or (length > 1 and np.any(keys[1:] <= keys[:-1])):
        raise ValueError("Corrupt compact coordinate ordering")
    yz = keys // shape[2]
    rows = np.column_stack((np.zeros(length, dtype=np.int32),
                            (yz // shape[1]).astype(np.int32),
                            (yz % shape[1]).astype(np.int32),
                            (keys % shape[2]).astype(np.int32)))
    return torch.from_numpy(rows)


def _pack_bool(values: torch.Tensor, length: int, codec: str, level: int) -> bytes:
    values = values.detach().to(device="cpu", dtype=torch.bool).numpy()
    if values.shape != (length,):
        raise ValueError("Invalid compact cache mask shape")
    return _compress(np.packbits(values, bitorder="little").tobytes(), codec, level)


def _unpack_bool(blob: bytes, length: int, codec: str) -> torch.Tensor:
    raw = _decompress(blob, codec)
    if len(raw) != (length + 7) // 8:
        raise ValueError("Corrupt compact mask payload")
    values = np.unpackbits(np.frombuffer(raw, dtype=np.uint8),
                           count=length, bitorder="little").astype(np.bool_)
    return torch.from_numpy(values)


def _pack_float(values: torch.Tensor, shape: tuple[int, ...],
                codec: str, level: int) -> tuple[str, bytes]:
    values = values.detach().to(device="cpu").contiguous()
    if tuple(values.shape) != shape or str(values.dtype) not in DTYPES:
        raise ValueError("Invalid compact cache floating tensor")
    return str(values.dtype), _compress(values.view(torch.uint8).numpy().tobytes(), codec, level)


def _unpack_float(dtype_name: str, blob: bytes, shape: tuple[int, ...],
                  codec: str) -> torch.Tensor:
    dtype = DTYPES.get(dtype_name)
    if dtype is None:
        raise ValueError("Invalid compact cache tensor dtype")
    raw = _decompress(blob, codec)
    if len(raw) != int(np.prod(shape)) * torch.empty((), dtype=dtype).element_size():
        raise ValueError("Corrupt compact floating payload")
    if not raw:
        return torch.empty(shape, dtype=dtype)
    return torch.from_numpy(np.frombuffer(raw, dtype=np.uint8).copy()).view(dtype).reshape(shape)


class CompactStudentCache(StudentValidationCache):
    """v3 cache with delta-packed coordinates, bit-packed masks, and lossless compression."""

    def __init__(self, root: Path, teacher_sha256: str, radar_variant: str,
                 *, split: str = "val", create: bool = False,
                 codec: str = "zlib", level: int = 6):
        if split not in ("train", "val"):
            raise ValueError("Student cache split must be train or val")
        if codec not in ("zlib", "zstd"):
            raise ValueError("Student cache codec must be zlib or zstd")
        if not (0 <= level <= 9 if codec == "zlib" else 1 <= level <= 22):
            raise ValueError("Invalid student cache compression level")
        self.root, self.codec, self.level = Path(root), codec, level
        source = hashlib.sha256()
        source.update(source_fingerprint().encode("ascii"))
        source.update(Path(__file__).read_bytes())
        expected = {"version": VERSION, "format": "packed-lossless",
                    "codec": codec, "level": level,
                    "teacher_sha256": teacher_sha256, "radar_variant": radar_variant,
                    "split": split, "augmentation_enabled": False,
                    "source_fingerprint": source.hexdigest(),
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
            raise ValueError("Compact student cache manifest differs from split, teacher, grid, codec, or source code")

    def path(self, frame_id: str) -> Path:
        if not frame_id.isdigit():
            raise ValueError("Expected a numeric VoD frame ID")
        return self.root / f"{frame_id}.ptz"

    def write_entry(self, frame_id: str, entry: dict) -> int:
        path = self.path(frame_id)
        if path.exists():
            self._read(frame_id)
            return path.stat().st_size
        candidate = entry["candidate_coords"]
        bottleneck = entry["bottleneck_coords"]
        n, m = len(candidate), len(bottleneck)
        feature_dtype, features = _pack_float(entry["teacher_features"], (m, 64),
                                              self.codec, self.level)
        evidence_dtype, evidence = _pack_float(entry["evidence"], (n,),
                                               self.codec, self.level)
        payload = {"version": VERSION, "frame_id": frame_id, "n": n, "m": m,
                   "candidate_delta": _pack_coords(candidate, GRID.shape_zyx, np.uint32,
                                                   self.codec, self.level),
                   "bottleneck_delta": _pack_coords(bottleneck, GRID.scale_shape(8),
                                                    np.uint16, self.codec, self.level),
                   "teacher_features": features, "teacher_features_dtype": feature_dtype,
                   "evidence": evidence, "evidence_dtype": evidence_dtype,
                   "occupied_bits": _pack_bool(entry["occupied"], n, self.codec, self.level),
                   "known_free_bits": _pack_bool(entry["known_free"], n, self.codec, self.level),
                   "clean_points_in_candidates": int(entry["clean_points_in_candidates"]),
                   "clean_points_in_grid": int(entry["clean_points_in_grid"])}
        temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
        try:
            torch.save(payload, temporary)
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)
        return path.stat().st_size

    def write(self, frame_id: str, student_output, teacher_output, targets,
              evidence: torch.Tensor) -> int:
        candidate = student_output.confidence_logits.coords
        bottleneck = student_output.conditioning.coords
        if not torch.equal(bottleneck, teacher_output.fused.coords):
            raise ValueError("Teacher and student bottleneck coordinates differ")
        return self.write_entry(frame_id, {
            "candidate_coords": candidate, "bottleneck_coords": bottleneck,
            "teacher_features": teacher_output.fused.features, "evidence": evidence,
            "occupied": targets.occupied, "known_free": targets.known_free,
            "clean_points_in_candidates": targets.clean_points_in_candidates,
            "clean_points_in_grid": targets.clean_points_in_grid})

    def _read(self, frame_id: str) -> dict:
        payload = torch.load(self.path(frame_id), map_location="cpu", weights_only=True)
        if payload.get("version") != VERSION or payload.get("frame_id") != frame_id:
            raise ValueError(f"Invalid compact student cache entry: {frame_id}")
        n, m = payload["n"], payload["m"]
        if not isinstance(n, int) or not isinstance(m, int) or n < 0 or m < 0:
            raise ValueError("Invalid compact student cache lengths")
        return {"version": VERSION, "frame_id": frame_id,
                "candidate_coords": _unpack_coords(payload["candidate_delta"], n,
                                                   GRID.shape_zyx, np.uint32, self.codec),
                "bottleneck_coords": _unpack_coords(payload["bottleneck_delta"], m,
                                                    GRID.scale_shape(8), np.uint16, self.codec),
                "teacher_features": _unpack_float(payload["teacher_features_dtype"],
                                                   payload["teacher_features"], (m, 64), self.codec),
                "evidence": _unpack_float(payload["evidence_dtype"], payload["evidence"],
                                          (n,), self.codec),
                "occupied": _unpack_bool(payload["occupied_bits"], n, self.codec),
                "known_free": _unpack_bool(payload["known_free_bits"], n, self.codec),
                "clean_points_in_candidates": payload["clean_points_in_candidates"],
                "clean_points_in_grid": payload["clean_points_in_grid"]}


def open_student_cache(root: Path, teacher_sha256: str, radar_variant: str,
                       *, split: str):
    """Read either on-disk version without migrating an active legacy cache."""
    manifest = json.loads((Path(root) / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("version") == VERSION and manifest.get("format") == "packed-lossless":
        return CompactStudentCache(root, teacher_sha256, radar_variant, split=split,
                                   codec=manifest["codec"], level=manifest["level"])
    return StudentValidationCache(root, teacher_sha256, radar_variant, split=split)

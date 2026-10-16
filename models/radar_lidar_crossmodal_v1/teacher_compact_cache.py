"""Lossless, versioned compression for frozen teacher input caches.

The v1 ``.pt`` format remains readable. This format only changes storage:
decoded tensors have the same shapes, dtypes, and bits as v1 entries.
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
from .teacher_static_cache import TeacherStaticCache, _source_hash


VERSION = 2
DTYPES = {str(dtype): dtype for dtype in (
    torch.float16, torch.bfloat16, torch.float32, torch.float64,
    torch.int16, torch.int32, torch.int64,
)}


def _compression_source_hash() -> str:
    digest = hashlib.sha256(_source_hash().encode("ascii"))
    digest.update(Path(__file__).read_bytes())
    return digest.hexdigest()


def _compress(raw: bytes, codec: str, level: int) -> bytes:
    if codec == "zlib":
        return zlib.compress(raw, level)
    if codec == "zstd":
        try:
            import zstandard
        except ImportError as exc:
            raise RuntimeError("zstd teacher cache requires pip install zstandard") from exc
        return zstandard.ZstdCompressor(level=level, write_checksum=True).compress(raw)
    raise ValueError(f"Unsupported teacher cache codec: {codec}")


def _decompress(blob: bytes, codec: str) -> bytes:
    if codec == "zlib":
        return zlib.decompress(blob)
    if codec == "zstd":
        try:
            import zstandard
        except ImportError as exc:
            raise RuntimeError("zstd teacher cache requires pip install zstandard") from exc
        return zstandard.ZstdDecompressor().decompress(blob)
    raise ValueError(f"Unsupported teacher cache codec: {codec}")


def _pack_bytes(raw: bytes, codec: str, level: int) -> dict:
    return {"length": len(raw), "sha256": hashlib.sha256(raw).hexdigest(),
            "data": _compress(raw, codec, level)}


def _unpack_bytes(packed: dict, codec: str) -> bytes:
    raw = _decompress(packed["data"], codec)
    if (len(raw) != packed["length"]
            or hashlib.sha256(raw).hexdigest() != packed["sha256"]):
        raise ValueError("Corrupt compact teacher cache payload")
    return raw


def _pack_tensor(value: torch.Tensor, codec: str, level: int,
                 *, byte_shuffle: bool = True) -> dict:
    value = value.detach().cpu().contiguous()
    if str(value.dtype) not in DTYPES:
        raise ValueError(f"Unsupported teacher cache tensor dtype: {value.dtype}")
    width = value.element_size()
    raw = value.view(torch.uint8).numpy().tobytes()
    if byte_shuffle and width > 1 and value.numel():
        raw = np.frombuffer(raw, dtype=np.uint8).reshape(-1, width).T.copy().tobytes()
    return {"dtype": str(value.dtype), "shape": list(value.shape),
            "byte_shuffle": byte_shuffle and width > 1,
            "payload": _pack_bytes(raw, codec, level)}


def _unpack_tensor(packed: dict, codec: str) -> torch.Tensor:
    dtype = DTYPES.get(packed["dtype"])
    if dtype is None:
        raise ValueError("Invalid compact teacher cache tensor dtype")
    shape = packed["shape"]
    if not isinstance(shape, list) or any(not isinstance(dim, int) or dim < 0 for dim in shape):
        raise ValueError("Invalid compact teacher cache tensor shape")
    raw = _unpack_bytes(packed["payload"], codec)
    width = torch.empty((), dtype=dtype).element_size()
    if len(raw) != int(np.prod(shape, dtype=np.int64)) * width:
        raise ValueError("Invalid compact teacher cache tensor length")
    if packed["byte_shuffle"] and width > 1 and raw:
        raw = np.frombuffer(raw, dtype=np.uint8).reshape(width, -1).T.copy().tobytes()
    if not raw:
        return torch.empty(shape, dtype=dtype)
    return torch.from_numpy(np.frombuffer(raw, dtype=np.uint8).copy()).view(dtype).reshape(shape)


def _pack_coords(coords: torch.Tensor, codec: str, level: int) -> dict:
    shape = GRID.scale_shape(8)
    rows = coords.detach().cpu().to(torch.int64).numpy()
    if (rows.ndim != 2 or rows.shape[1] != 4 or np.any(rows[:, 0] != 0)
            or np.any(rows[:, 1:] < 0) or np.any(rows[:, 1:] >= np.asarray(shape))):
        raise ValueError("Teacher coordinates must be in the single-frame stride-8 grid")
    keys = (rows[:, 1] * shape[1] + rows[:, 2]) * shape[2] + rows[:, 3]
    if (len(keys) > 1 and np.any(keys[1:] <= keys[:-1])) or np.prod(shape) >= 2**32:
        raise ValueError("Teacher coordinates must be sorted and fit uint32")
    delta = np.empty(len(keys), dtype=np.uint32)
    if len(keys):
        delta[0] = keys[0]
        delta[1:] = np.diff(keys).astype(np.uint32)
    return {"count": len(keys), "payload": _pack_bytes(delta.tobytes(), codec, level)}


def _unpack_coords(packed: dict, codec: str) -> torch.Tensor:
    count = packed["count"]
    raw = _unpack_bytes(packed["payload"], codec)
    if not isinstance(count, int) or count < 0 or len(raw) != count * 4:
        raise ValueError("Invalid compact teacher coordinate length")
    keys = np.cumsum(np.frombuffer(raw, dtype=np.uint32), dtype=np.uint64)
    z, y, x = GRID.scale_shape(8)
    if np.any(keys >= z * y * x) or (count > 1 and np.any(keys[1:] <= keys[:-1])):
        raise ValueError("Invalid compact teacher coordinate order")
    rows = np.column_stack((np.zeros(count, dtype=np.int32),
                            (keys // (y * x)).astype(np.int32),
                            ((keys // x) % y).astype(np.int32),
                            (keys % x).astype(np.int32)))
    return torch.from_numpy(rows)


def _pack_neighbors(indices: torch.Tensor, lidar_count: int,
                    codec: str, level: int) -> dict:
    rows = indices.detach().cpu().to(torch.int64).numpy()
    if (rows.ndim != 2 or lidar_count >= 2**32
            or np.any(rows < -1) or np.any(rows >= lidar_count)):
        raise ValueError("Invalid teacher neighbor indices")
    dtype = np.uint16 if lidar_count <= 65535 else np.uint32
    sentinel = np.iinfo(dtype).max
    encoded = np.where(rows < 0, sentinel, rows).astype(dtype)
    return {"shape": list(rows.shape), "dtype": np.dtype(dtype).name,
            "payload": _pack_bytes(encoded.tobytes(), codec, level)}


def _unpack_neighbors(packed: dict, lidar_count: int, codec: str) -> torch.Tensor:
    shape = packed["shape"]
    if (not isinstance(shape, list) or len(shape) != 2
            or any(not isinstance(dim, int) or dim < 0 for dim in shape)
            or packed["dtype"] not in ("uint16", "uint32")):
        raise ValueError("Invalid compact teacher neighbor shape")
    dtype = np.dtype(packed["dtype"])
    raw = _unpack_bytes(packed["payload"], codec)
    if len(raw) != shape[0] * shape[1] * dtype.itemsize:
        raise ValueError("Invalid compact teacher neighbor length")
    values = np.frombuffer(raw, dtype=dtype).astype(np.int64).reshape(shape)
    values[values == np.iinfo(dtype).max] = -1
    if np.any(values >= lidar_count):
        raise ValueError("Compact teacher neighbor refers to a missing LiDAR site")
    return torch.from_numpy(values.astype(np.int32))


def _same_entry(left: dict, right: dict) -> bool:
    if left.keys() != right.keys():
        return False
    for key, value in left.items():
        other = right[key]
        if isinstance(value, torch.Tensor):
            if (value.dtype != other.dtype or value.shape != other.shape
                    or not torch.equal(value.contiguous().view(torch.uint8),
                                       other.contiguous().view(torch.uint8))):
                return False
        elif value != other:
            return False
    return True


class CompactTeacherCache(TeacherStaticCache):
    """Compact cache with the same ``load_batch`` contract as the v1 cache."""

    def __init__(self, root: Path, *, split: str, radar_variant: str,
                 radar_sha256: str, lidar_sha256: str, radius_m: float,
                 max_candidates: int, create: bool = False,
                 codec: str = "zlib", level: int = 9):
        if split not in ("train", "val") or codec not in ("zlib", "zstd"):
            raise ValueError("Invalid compact teacher cache split or codec")
        if not (0 <= level <= 9 if codec == "zlib" else 1 <= level <= 22):
            raise ValueError("Invalid compact teacher cache compression level")
        self.root, self.codec, self.level = Path(root) / split, codec, level
        expected = {"version": VERSION, "format": "packed-lossless",
                    "codec": codec, "level": level, "split": split,
                    "radar_variant": radar_variant, "radar_sha256": radar_sha256,
                    "lidar_sha256": lidar_sha256, "radius_m": radius_m,
                    "max_candidates": max_candidates,
                    "source_hash": _compression_source_hash(),
                    "augmentation_enabled": False,
                    "grid_minimum_xyz": list(GRID.minimum_xyz),
                    "grid_size_xyz": list(GRID.size_xyz)}
        manifest = self.root / "manifest.json"
        if create:
            self.root.mkdir(parents=True, exist_ok=True)
            if not manifest.exists():
                temporary = manifest.with_suffix(".json.tmp")
                temporary.write_text(json.dumps(expected), encoding="utf-8")
                temporary.replace(manifest)
        if not manifest.is_file() or json.loads(manifest.read_text(encoding="utf-8")) != expected:
            raise ValueError("Compact teacher cache manifest differs from encoders, geometry, or source")

    def path(self, frame_id: str) -> Path:
        if not frame_id.isdigit():
            raise ValueError("Expected numeric VoD frame ID")
        return self.root / f"{frame_id}.ptz"

    def write_entry(self, frame_id: str, entry: dict) -> int:
        path = self.path(frame_id)
        if path.exists():
            if not _same_entry(entry, self._read(frame_id)):
                raise ValueError(f"Existing compact teacher entry differs: {frame_id}")
            return path.stat().st_size
        r, l = entry["radar_coords"], entry["lidar_coords"]
        if (entry.get("version") != 1 or entry.get("frame_id") != frame_id
                or r.shape != (len(r), 4) or l.shape != (len(l), 4)
                or entry["radar_features"].shape != (len(r), 64)
                or entry["lidar_features"].shape != (len(l), 64)
                or entry["neighbor_indices"].shape[0] != len(r)
                or entry["neighbor_distances"].shape != entry["neighbor_indices"].shape
                or entry["radar_ids"].shape != (len(r),)
                or entry["lidar_ids"].shape != (len(l),)):
            raise ValueError("Invalid teacher cache entry for lossless conversion")
        payload = {"version": VERSION, "frame_id": frame_id,
                   "radar_coords": _pack_coords(r, self.codec, self.level),
                   "lidar_coords": _pack_coords(l, self.codec, self.level),
                   "radar_features": _pack_tensor(entry["radar_features"], self.codec, self.level),
                   "lidar_features": _pack_tensor(entry["lidar_features"], self.codec, self.level),
                   "neighbor_indices": _pack_neighbors(entry["neighbor_indices"], len(l),
                                                       self.codec, self.level),
                   "neighbor_distances": _pack_tensor(entry["neighbor_distances"],
                                                      self.codec, self.level),
                   "radar_ids": _pack_tensor(entry["radar_ids"], self.codec, self.level),
                   "lidar_ids": _pack_tensor(entry["lidar_ids"], self.codec, self.level),
                   "classes": entry["classes"]}
        temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
        try:
            torch.save(payload, temporary)
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)
        if not _same_entry(entry, self._read(frame_id)):
            raise ValueError(f"Compact teacher round-trip differs: {frame_id}")
        return path.stat().st_size

    def write(self, frame_id: str, radar, lidar, neighbors, radar_ids,
              lidar_ids, classes: dict[int, str]) -> int:
        entry = {"version": 1, "frame_id": frame_id,
                 "radar_coords": radar.coords.to("cpu", dtype=torch.int32),
                 "radar_features": radar.features.detach().cpu(),
                 "lidar_coords": lidar.coords.to("cpu", dtype=torch.int32),
                 "lidar_features": lidar.features.detach().cpu(),
                 "neighbor_indices": neighbors.indices.to("cpu", dtype=torch.int32),
                 "neighbor_distances": neighbors.distances_m.cpu(),
                 "radar_ids": radar_ids.cpu(), "lidar_ids": lidar_ids.cpu(),
                 "classes": classes}
        return self.write_entry(frame_id, entry)

    def _read(self, frame_id: str) -> dict:
        packed = torch.load(self.path(frame_id), map_location="cpu", weights_only=True)
        if packed.get("version") != VERSION or packed.get("frame_id") != frame_id:
            raise ValueError(f"Invalid compact teacher cache entry: {frame_id}")
        entry = {"version": 1, "frame_id": frame_id,
                 "radar_coords": _unpack_coords(packed["radar_coords"], self.codec),
                 "lidar_coords": _unpack_coords(packed["lidar_coords"], self.codec),
                 "radar_features": _unpack_tensor(packed["radar_features"], self.codec),
                 "lidar_features": _unpack_tensor(packed["lidar_features"], self.codec),
                 "neighbor_indices": None, "neighbor_distances": None,
                 "radar_ids": _unpack_tensor(packed["radar_ids"], self.codec),
                 "lidar_ids": _unpack_tensor(packed["lidar_ids"], self.codec),
                 "classes": packed["classes"]}
        entry["neighbor_indices"] = _unpack_neighbors(
            packed["neighbor_indices"], len(entry["lidar_coords"]), self.codec)
        entry["neighbor_distances"] = _unpack_tensor(packed["neighbor_distances"], self.codec)
        if (entry["radar_features"].shape != (len(entry["radar_coords"]), 64)
                or entry["lidar_features"].shape != (len(entry["lidar_coords"]), 64)
                or entry["neighbor_indices"].shape[0] != len(entry["radar_coords"])
                or entry["neighbor_distances"].shape != entry["neighbor_indices"].shape
                or entry["radar_ids"].shape != (len(entry["radar_coords"]),)
                or entry["lidar_ids"].shape != (len(entry["lidar_coords"]),)):
            raise ValueError(f"Invalid compact teacher cache shapes: {frame_id}")
        return entry


def open_teacher_cache(root: Path, *, split: str, radar_variant: str,
                       radar_sha256: str, lidar_sha256: str, radius_m: float,
                       max_candidates: int, create: bool = False,
                       cache_format: str = "auto", codec: str = "zlib", level: int = 9):
    """Read either cache version; create compact entries for new cache roots."""
    manifest = Path(root) / split / "manifest.json"
    requested_format = cache_format
    if cache_format == "auto":
        if manifest.is_file():
            cache_format = ("compact" if json.loads(manifest.read_text(encoding="utf-8"))
                            .get("version") == VERSION else "legacy")
        else:
            cache_format = "compact"
    common = dict(split=split, radar_variant=radar_variant,
                  radar_sha256=radar_sha256, lidar_sha256=lidar_sha256,
                  radius_m=radius_m, max_candidates=max_candidates, create=create)
    if cache_format == "compact":
        if manifest.is_file():
            saved = json.loads(manifest.read_text(encoding="utf-8"))
            if (requested_format == "compact" and
                    (codec != saved.get("codec") or level != saved.get("level"))):
                raise ValueError("Existing compact teacher cache uses another codec or level")
            codec, level = saved.get("codec", codec), saved.get("level", level)
        return CompactTeacherCache(root, codec=codec, level=level, **common)
    if cache_format == "legacy":
        return TeacherStaticCache(root, **common)
    raise ValueError("Teacher cache format must be auto, compact, or legacy")

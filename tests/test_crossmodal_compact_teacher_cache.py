"""Lossless teacher cache conversion and training compatibility."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from models.radar_lidar_stage1.correspondence import Neighborhood
from models.radar_lidar_stage1.sparse import SparseSites
from models.radar_lidar_crossmodal_v1.encoder import GRID
from models.radar_lidar_crossmodal_v1.teacher import BottleneckTeacherCore, TeacherConfig
from models.radar_lidar_crossmodal_v1.teacher_compact_cache import (
    CompactTeacherCache, _pack_neighbors, _pack_tensor, _same_entry,
    _unpack_neighbors, _unpack_tensor,
    open_teacher_cache,
)
from models.radar_lidar_crossmodal_v1.teacher_static_cache import TeacherStaticCache
from models.radar_lidar_crossmodal_v1.train_teacher import _epoch
from scripts.compact_crossmodal_teacher_cache import convert_split


COMMON = dict(split="train", radar_variant="radar-test",
              radar_sha256="radar-hash", lidar_sha256="lidar-hash",
              radius_m=6., max_candidates=3)


def _sites(coords, features):
    return SparseSites(torch.tensor(coords, dtype=torch.long), features,
                       GRID.scale_shape(8), 8)


def _write_legacy(root, frame_id="00001"):
    cache = TeacherStaticCache(root, **COMMON, create=True)
    radar = _sites([[0, 1, 4, 4], [0, 1, 4, 6]],
                   torch.arange(128, dtype=torch.float32).reshape(2, 64) / 9)
    radar.features[0, 0] = -0.0
    lidar = _sites([[0, 1, 4, 4], [0, 1, 4, 5], [0, 1, 4, 6]],
                   torch.arange(192, dtype=torch.float32).reshape(3, 64) / 11)
    neighbors = Neighborhood(torch.tensor([[0, 1, -1], [2, 1, -1]]),
                             torch.tensor([[0., .2, float("inf")],
                                           [.4, .2, float("inf")]]))
    cache.write(frame_id, radar, lidar, neighbors,
                torch.tensor([7, -1]), torch.tensor([7, 8, -1]),
                {7: "Car", 8: "Pedestrian"})
    return cache


@pytest.mark.parametrize("codec,level", [("zlib", 9), ("zstd", 9)])
def test_compact_teacher_cache_is_bit_exact_and_rebases_batch_indices(
        tmp_path, codec, level):
    if codec == "zstd":
        pytest.importorskip("zstandard")
    legacy = _write_legacy(tmp_path / "legacy")
    _write_legacy(tmp_path / "legacy", "00002")
    compact = CompactTeacherCache(tmp_path / "compact", **COMMON, create=True,
                                  codec=codec, level=level)
    for frame in ("00001", "00002"):
        entry = legacy._read(frame)
        if frame == "00001":
            compact.write(frame, *legacy.load_batch([frame], "cpu"))
        else:
            compact.write_entry(frame, entry)
        assert _same_entry(entry, compact._read(frame))
        assert compact.path(frame).suffix == ".ptz"
    opened = open_teacher_cache(tmp_path / "compact", **COMMON)
    assert isinstance(opened, CompactTeacherCache)
    loaded = opened.load_batch(["00001", "00002"], "cpu")
    original = legacy.load_batch(["00001", "00002"], "cpu")
    for a, b in zip(loaded[:5], original[:5]):
        if isinstance(a, SparseSites):
            assert torch.equal(a.coords, b.coords)
            assert torch.equal(a.features.view(torch.uint8), b.features.view(torch.uint8))
        elif isinstance(a, Neighborhood):
            assert torch.equal(a.indices, b.indices)
            assert torch.equal(a.distances_m.view(torch.uint8),
                               b.distances_m.view(torch.uint8))
        else:
            assert torch.equal(a, b)
    assert loaded[5] == original[5]

    class CachedTeacher(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.config = TeacherConfig(max_candidates=3)
            self.core = BottleneckTeacherCore(self.config)

        def forward(self, *args):
            raise AssertionError("Compact cache must bypass frozen encoders")

    model = CachedTeacher()
    optimizer = torch.optim.SGD(model.core.parameters(), lr=.01)
    report = _epoch(model, SimpleNamespace(set_epoch=lambda epoch: None),
                    [{"frame_id": ["00001", "00002"]}], "cpu",
                    optimizer=optimizer, epoch=2, static_cache=opened)
    assert report["frames"] == 2
    assert report["radar_active_sites"] == 4


def test_conversion_resumes_without_changing_legacy_cache(tmp_path):
    legacy = _write_legacy(tmp_path / "legacy")
    original = legacy.path("00001").read_bytes()
    first = convert_split(tmp_path / "legacy", tmp_path / "compact",
                          "train", "zlib", 9, None, 0.)
    second = convert_split(tmp_path / "legacy", tmp_path / "compact",
                           "train", "zlib", 9, None, 0.)
    assert first["frames"] == second["frames"] == 1
    assert first["compact_bytes"] == second["compact_bytes"]
    assert legacy.path("00001").read_bytes() == original
    assert _same_entry(legacy._read("00001"),
                       CompactTeacherCache(tmp_path / "compact", **COMMON)._read("00001"))
    assert isinstance(open_teacher_cache(tmp_path / "legacy", **COMMON),
                      TeacherStaticCache)


def test_compact_teacher_detects_corruption_and_source_mismatch(tmp_path):
    legacy = _write_legacy(tmp_path / "legacy")
    assert isinstance(open_teacher_cache(tmp_path / "fresh", **COMMON, create=True),
                      CompactTeacherCache)
    compact = CompactTeacherCache(tmp_path / "compact", **COMMON, create=True)
    compact.write_entry("00001", legacy._read("00001"))
    packed = torch.load(compact.path("00001"), weights_only=True)
    packed["radar_features"]["payload"]["sha256"] = "0" * 64
    torch.save(packed, compact.path("00001"))
    with pytest.raises(ValueError, match="Corrupt"):
        compact._read("00001")
    with pytest.raises(ValueError, match="manifest"):
        CompactTeacherCache(tmp_path / "compact", **{**COMMON,
                            "radar_sha256": "different"})


def test_neighbor_index_width_grows_losslessly_when_needed():
    indices = torch.tensor([[69999, -1, 0]], dtype=torch.int32)
    packed = _pack_neighbors(indices, 70000, "zlib", 9)
    assert packed["dtype"] == "uint32"
    assert torch.equal(_unpack_neighbors(packed, 70000, "zlib"), indices)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16,
                                   torch.float32, torch.float64, torch.int64])
def test_tensor_byte_shuffle_preserves_dtype_and_bits(dtype):
    values = (torch.tensor([-0., 1.5, -3.25], dtype=dtype)
              if dtype != torch.int64 else torch.tensor([-1, 0, 2**40], dtype=dtype))
    packed = _pack_tensor(values, "zlib", 9)
    decoded = _unpack_tensor(packed, "zlib")
    assert decoded.dtype == dtype
    assert torch.equal(decoded.view(torch.uint8), values.view(torch.uint8))

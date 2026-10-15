"""Exact v3 student-cache round trips and legacy compatibility."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from models.radar_lidar_crossmodal_v1.encoder import GRID
from models.radar_lidar_crossmodal_v1.student import StudentOutput
from models.radar_lidar_crossmodal_v1.student_compact_cache import (
    CompactStudentCache, open_student_cache,
)
from models.radar_lidar_crossmodal_v1.student_validation_cache import (
    CachedTargets, StudentValidationCache,
)
from models.radar_lidar_stage1.sparse import SparseSites


def _sites(coords, features, stride):
    return SparseSites(torch.tensor(coords, dtype=torch.long), features,
                       GRID.scale_shape(stride), stride)


def _frame(batch=0, offset=0):
    bottleneck = _sites([[batch, 0, 1, 1], [batch, 1, 2, 3]],
                        torch.ones((2, 64)), 8)
    fine = _sites([[batch, 0, 1, 1], [batch, 0, 1, 2],
                   [batch, 15, 511, 511]], torch.tensor([[1.], [-1.], [0.]]), 1)
    output = StudentOutput(bottleneck, fine)
    teacher = SimpleNamespace(fused=bottleneck.replace_features(
        torch.arange(128, dtype=torch.float32).reshape(2, 64) + offset))
    targets = CachedTargets(torch.tensor([True, False, False]),
                            torch.tensor([False, True, False]), 7, 12)
    return output, teacher, targets, torch.tensor([0.25, 0.5, 0.0])


def test_compact_cache_is_lossless_and_keeps_batches_separate(tmp_path):
    cache = CompactStudentCache(tmp_path, "teacher-hash", "radar-test", create=True)
    for name, offset in (("00001", 0), ("00002", 100)):
        cache.write(name, *_frame(offset=offset))
        saved = cache._read(name)
        output, teacher, targets, evidence = _frame(offset=offset)
        assert torch.equal(saved["candidate_coords"], output.confidence_logits.coords.int())
        assert torch.equal(saved["bottleneck_coords"], output.conditioning.coords.int())
        assert torch.equal(saved["teacher_features"], teacher.fused.features)
        assert torch.equal(saved["evidence"], evidence)
        assert torch.equal(saved["occupied"], targets.occupied)
        assert torch.equal(saved["known_free"], targets.known_free)
    assert isinstance(open_student_cache(tmp_path, "teacher-hash", "radar-test", split="val"),
                      CompactStudentCache)
    out0, _, _, _ = _frame(batch=0)
    out1, _, _, _ = _frame(batch=1)
    joined = StudentOutput(
        _sites(torch.cat((out0.conditioning.coords, out1.conditioning.coords)).tolist(),
               torch.ones((4, 64)), 8),
        _sites(torch.cat((out0.confidence_logits.coords,
                          out1.confidence_logits.coords)).tolist(),
               torch.ones((6, 1)), 1))
    teacher, targets, evidence = cache.load_batch(["00001", "00002"], joined, "cpu")
    assert teacher.coords[:, 0].tolist() == [0, 0, 1, 1]
    assert teacher.features[2, 0].item() == 100
    assert targets.occupied.tolist() == [True, False, False] * 2
    assert evidence.tolist() == pytest.approx([.25, .5, 0.] * 2)
    with pytest.raises(ValueError, match="manifest"):
        CompactStudentCache(tmp_path, "wrong", "radar-test")


def test_legacy_cache_stays_readable(tmp_path):
    cache = StudentValidationCache(tmp_path, "teacher-hash", "radar-test", create=True)
    cache.write("00001", *_frame())
    selected = open_student_cache(tmp_path, "teacher-hash", "radar-test", split="val")
    assert isinstance(selected, StudentValidationCache)
    assert torch.equal(selected._read("00001")["evidence"], torch.tensor([.25, .5, 0.]))


def test_compact_cache_empty_frame_and_float_bits(tmp_path):
    cache = CompactStudentCache(tmp_path, "teacher-hash", "radar-test", create=True)
    entry = {"candidate_coords": torch.empty((0, 4), dtype=torch.int32),
             "bottleneck_coords": torch.empty((0, 4), dtype=torch.int32),
             "teacher_features": torch.empty((0, 64), dtype=torch.bfloat16),
             "evidence": torch.empty(0, dtype=torch.float32),
             "occupied": torch.empty(0, dtype=torch.bool),
             "known_free": torch.empty(0, dtype=torch.bool),
             "clean_points_in_candidates": 0, "clean_points_in_grid": 0}
    cache.write_entry("00001", entry)
    for key in ("candidate_coords", "bottleneck_coords", "teacher_features",
                "evidence", "occupied", "known_free"):
        assert torch.equal(cache._read("00001")[key], entry[key])
    entry = dict(entry)
    entry.update(candidate_coords=torch.tensor([[0, 15, 511, 511]], dtype=torch.int32),
                 bottleneck_coords=torch.tensor([[0, 1, 63, 63]], dtype=torch.int32),
                 teacher_features=torch.arange(64, dtype=torch.float32).reshape(1, 64),
                 evidence=torch.tensor([0.123456789], dtype=torch.float32),
                 occupied=torch.tensor([True]), known_free=torch.tensor([False]))
    cache.write_entry("00002", entry)
    restored = cache._read("00002")
    for key in ("candidate_coords", "bottleneck_coords", "teacher_features",
                "evidence", "occupied", "known_free"):
        assert torch.equal(restored[key], entry[key])


def test_packed_coordinates_and_masks_save_space(tmp_path):
    legacy = StudentValidationCache(tmp_path / "legacy", "teacher-hash", "radar-test",
                                    create=True)
    compact = CompactStudentCache(tmp_path / "compact", "teacher-hash", "radar-test",
                                  create=True)
    shape = GRID.shape_zyx
    keys = np.arange(100_000, dtype=np.int64)
    yz, x = np.divmod(keys, shape[2])
    z, y = np.divmod(yz, shape[1])
    coords = torch.from_numpy(np.column_stack((np.zeros(len(keys)), z, y, x))).long()
    bottleneck = torch.tensor([[0, 0, 1, 1], [0, 1, 2, 3]], dtype=torch.long)
    output = StudentOutput(_sites(bottleneck.tolist(), torch.ones((2, 64)), 8),
                           _sites(coords.tolist(), torch.ones((len(keys), 1)), 1))
    teacher = SimpleNamespace(fused=output.conditioning.replace_features(torch.ones((2, 64))))
    targets = CachedTargets(torch.arange(len(keys)) % 100 == 0,
                            torch.arange(len(keys)) % 100 == 1, 1000, 100_000)
    evidence = torch.arange(len(keys), dtype=torch.float32) / len(keys)
    old_bytes = legacy.write("00001", output, teacher, targets, evidence)
    new_bytes = compact.write("00001", output, teacher, targets, evidence)
    assert new_bytes < old_bytes * 0.7
    assert torch.equal(compact._read("00001")["evidence"], evidence)

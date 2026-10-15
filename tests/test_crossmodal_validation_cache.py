"""Focused checks for batched validation and reusable teacher supervision."""

from types import SimpleNamespace

import pytest
import torch

from models.radar_lidar_crossmodal_v1.encoder import GRID
from models.radar_lidar_crossmodal_v1.student import StudentConfig, StudentOutput
from models.radar_lidar_crossmodal_v1.student_loss import (
    radar_supported_iou, radar_supported_iou_sweep, student_objective,
)
from models.radar_lidar_crossmodal_v1.student_validation_cache import (
    CachedTargets, StudentValidationCache,
)
from models.radar_lidar_crossmodal_v1.teacher import tiled_local_neighbors
from models.radar_lidar_stage1.correspondence import local_neighbors
from models.radar_lidar_stage1.sparse import SparseSites


def _sites(coords, features, stride):
    return SparseSites(torch.tensor(coords, dtype=torch.long), features,
                       GRID.scale_shape(stride), stride)


def _output(batch=0, logits=(2., -2., 2.)):
    conditioning = _sites([[batch, 0, 1, 1]], torch.ones((1, 64)), 8)
    confidence = _sites([[batch, 0, 1, index] for index in range(1, len(logits) + 1)],
                        torch.tensor(logits)[:, None], 1)
    return StudentOutput(conditioning, confidence)


def _targets():
    return CachedTargets(torch.tensor([True, False, False]),
                         torch.tensor([False, True, False]), 4, 10)


def test_gpu_style_threshold_sweep_matches_existing_counts():
    output = _output()
    targets = _targets()
    evidence = torch.tensor([.8, .6, 0.])
    thresholds = (.1, .5, .9)
    counts = radar_supported_iou_sweep(output, targets, thresholds, evidence, .2)
    assert counts.shape == (3, 3, 2)
    for row, threshold in zip(counts.tolist(), thresholds):
        reference = radar_supported_iou(output, targets, threshold, evidence, .2)
        for values, key in zip(row, ("known_space_masked", "conventional_clean_scan",
                                    "teacher_correspondence_supported")):
            assert values == [reference[key]["intersection"], reference[key]["union"]]


def test_tiled_spatial_query_matches_kdtree_without_cross_batch_neighbors():
    radar = _sites([[0, 0, 2, 2], [0, 0, 4, 4], [1, 0, 2, 2]],
                   torch.zeros((3, 64)), 8)
    lidar = _sites([[0, 0, 2, 2], [0, 0, 3, 2], [1, 0, 2, 2], [1, 0, 4, 2]],
                   torch.zeros((4, 64)), 8)
    expected = local_neighbors(radar, lidar, GRID, 3.0, 3)
    actual = tiled_local_neighbors(radar, lidar, 3.0, 3, max_pairs=4)
    torch.testing.assert_close(actual.indices, expected.indices)
    torch.testing.assert_close(actual.distances_m, expected.distances_m,
                               atol=1e-5, rtol=1e-5)
    assert actual.indices[2, 0] == 2


def test_validation_cache_joins_frames_without_crossing_sample_boundaries(tmp_path):
    cache = StudentValidationCache(tmp_path, "teacher-hash", "radar-test", create=True)
    first, second = _output(logits=(2.,)), _output(logits=(-2.,))
    for frame_id, output, occupied, evidence in (
        ("00001", first, True, .8), ("00002", second, False, .3)
    ):
        teacher = SimpleNamespace(fused=output.conditioning.replace_features(
            torch.full((1, 64), float(int(frame_id)))))
        targets = CachedTargets(torch.tensor([occupied]), torch.tensor([not occupied]),
                                int(occupied), 2)
        cache.write(frame_id, output, teacher, targets, torch.tensor([evidence]))
    batched = StudentOutput(
        _sites([[0, 0, 1, 1], [1, 0, 1, 1]], torch.ones((2, 64)), 8),
        _sites([[0, 0, 1, 1], [1, 0, 1, 1]], torch.tensor([[2.], [-2.]]), 1))
    teacher_sites, targets, evidence = cache.load_batch(["00001", "00002"],
                                                        batched, "cpu")
    assert targets.occupied.tolist() == [True, False]
    assert targets.known_free.tolist() == [False, True]
    assert evidence.tolist() == pytest.approx([.8, .3])
    assert teacher_sites.coords[:, 0].tolist() == [0, 1]
    assert teacher_sites.features[:, 0].tolist() == [1., 2.]
    assert targets.clean_points_in_candidates == 1
    assert targets.clean_points_in_grid == 4
    total, _ = student_objective(batched, teacher_sites, targets, StudentConfig(), evidence)
    assert torch.isfinite(total)
    wrong = StudentOutput(batched.conditioning,
                          _sites([[0, 0, 1, 2], [1, 0, 1, 1]],
                                 torch.tensor([[2.], [-2.]]), 1))
    with pytest.raises(ValueError, match="coordinates differ"):
        cache.load_batch(["00001", "00002"], wrong, "cpu")
    with pytest.raises(ValueError, match="manifest"):
        StudentValidationCache(tmp_path, "wrong-teacher", "radar-test")

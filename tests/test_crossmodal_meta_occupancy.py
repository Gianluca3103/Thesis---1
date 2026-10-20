"""Focused CPU checks for the new supervision mask and bilevel gradient path."""

from types import SimpleNamespace

import torch
from torch import nn

from models.radar_lidar_stage1.sparse import SparseSites
from models.radar_lidar_stage1.correspondence import Neighborhood
from models.radar_lidar_crossmodal_v1.encoder import GRID
from models.radar_lidar_crossmodal_v1.meta_occupancy import (
    MetaOccupancyConfig, candidate_support, feature_distillation,
    meta_teacher_step, occupancy_loss, scan_overlap_counts,
    scan_overlap_counts_observed, scan_overlap_rates)
from models.radar_lidar_crossmodal_v1.meta_cache import (
    CachedOccupancyTargets, MetaOccupancyCache, collate_meta_cache,
    prepare_meta_batch)
from models.radar_lidar_crossmodal_v1.student import RadarOnlyStudent, StudentOutput
from models.radar_lidar_crossmodal_v1.teacher import BottleneckTeacherCore, TeacherConfig


def test_support_is_exact_radius_and_keeps_batches_separate():
    coords = torch.tensor([[0, 8, 256, 256], [0, 8, 256, 300],
                           [1, 8, 256, 256]], dtype=torch.long)
    radar = torch.zeros(2, 1, 7)
    radar[1, 0, 0] = 5.0
    radar[1, 0, 1] = 5.0
    valid = torch.ones(2, 1, dtype=torch.bool)
    support = candidate_support(coords, radar, valid, 3.2)
    assert support.tolist() == [True, False, False]


def test_occupancy_loss_ignores_unknown_and_outside_radius():
    logits = torch.tensor([[1.0], [9.0], [-9.0]], requires_grad=True)
    coords = torch.tensor([[0, 8, 256, 256], [0, 8, 256, 257],
                           [0, 8, 256, 258]], dtype=torch.long)
    sites = SparseSites(coords, logits, GRID.shape_zyx)
    output = StudentOutput(sites, sites)
    targets = SimpleNamespace(occupied=torch.tensor([True, False, False]),
                              known_free=torch.tensor([False, False, True]),
                              clean_point_count=torch.tensor([1, 0, 0]))
    loss, stats = occupancy_loss(output, targets,
                                 torch.tensor([True, True, False]), 0.5)
    assert stats["known"] == 1
    assert stats["represented_clean_points"] == 1
    loss.backward()
    assert logits.grad[0].abs().item() > 0
    assert logits.grad[1].item() == 0
    assert logits.grad[2].item() == 0


def test_conditioned_student_decoder_cannot_use_radar_skip_pyramid():
    class Decoder(nn.Module):
        def forward(self, feature):
            self.received = feature
            return feature

    student = RadarOnlyStudent.__new__(RadarOnlyStudent)
    nn.Module.__init__(student)
    student.config = SimpleNamespace(decoder_from_conditioned=True)
    student.confidence_decoder = Decoder()
    feature, shared = object(), object()
    student.encode_conditioning = lambda *args: (feature, shared, ("radar_skip",))
    output = student(torch.empty(0), torch.empty(0))
    assert student.confidence_decoder.received is feature
    assert output.conditioning is feature


def test_exact_and_0_2m_scan_matches_count_unrepresented_lidar_voxels():
    predicted_coords = torch.tensor([[0, 8, 256, 256], [0, 8, 256, 261],
                                     [0, 8, 256, 262]])
    observed_coords = torch.tensor([[0, 8, 256, 256], [0, 8, 256, 260]])
    sites = SparseSites(predicted_coords, torch.full((3, 1), 10.0), GRID.shape_zyx)
    clean_xyz = GRID.centers_xyz(observed_coords)
    clean = torch.cat((clean_xyz, torch.ones(2, 1)), dim=1)[None]
    radar = torch.zeros(1, 1, 7)
    valid = torch.ones(1, 1, dtype=torch.bool)
    support = candidate_support(predicted_coords, radar, valid, 3.2)
    result = scan_overlap_counts(StudentOutput(sites, sites), clean,
                                 torch.ones(1, 2, dtype=torch.bool),
                                 radar, valid, support, 0.5, 3.2)
    assert result == {"predicted": 3, "observed": 2,
                      "tp_exact": 1, "tp_at_0_2m": 2}
    assert scan_overlap_counts_observed(predicted_coords, observed_coords) == result
    rates = scan_overlap_rates(result)
    assert rates["iou_exact"] == 0.25
    assert rates["f1_exact"] == 0.4
    assert rates["iou_at_0_2m"] == 2 / 3
    assert rates["f1_at_0_2m"] == 0.8
    assert scan_overlap_rates({"predicted": 0, "observed": 0,
                               "tp_exact": 0, "tp_at_0_2m": 0})["iou_exact"] is None


def test_tolerant_scan_matching_is_one_to_one_and_frame_local():
    predicted_coords = torch.tensor([[0, 8, 256, 259], [0, 8, 256, 261]])
    observed_coords = torch.tensor([[0, 8, 256, 260]])
    sites = SparseSites(predicted_coords, torch.full((2, 1), 10.0), GRID.shape_zyx)
    center = GRID.centers_xyz(observed_coords)
    clean = torch.cat((center, torch.ones(1, 1)), dim=1)[None]
    radar = torch.zeros(1, 1, 7)
    valid = torch.ones(1, 1, dtype=torch.bool)
    support = candidate_support(predicted_coords, radar, valid, 3.2)
    result = scan_overlap_counts(StudentOutput(sites, sites), clean,
                                 torch.ones(1, 1, dtype=torch.bool),
                                 radar, valid, support, 0.5, 3.2)
    assert result["tp_exact"] == 0
    assert result["tp_at_0_2m"] == 1

    clean_two = torch.zeros(2, 1, 4)
    clean_two[1, 0, :3] = center[0]
    radar_two = torch.zeros(2, 1, 7)
    valid_two = torch.ones(2, 1, dtype=torch.bool)
    result = scan_overlap_counts(StudentOutput(sites, sites), clean_two,
                                 torch.tensor([[False], [True]]), radar_two,
                                 valid_two, support, 0.5, 3.2)
    assert result["tp_at_0_2m"] == 0


class ToyTeacher(nn.Module):
    def __init__(self):
        super().__init__()
        self.core = nn.Linear(1, 1, bias=False)
        self.core.weight.data.fill_(0.8)

    def forward(self, radar, radar_valid, clean, clean_valid):
        coords = torch.tensor([[0, 1, 32, 32]], device=radar.device)
        lidar = SparseSites(coords, clean[:, 0, :1], GRID.scale_shape(8), 8)
        fused = lidar.replace_features(self.core(lidar.features))
        return SimpleNamespace(
            fused=fused, lidar_frozen=lidar,
            neighbors=SimpleNamespace(valid=torch.ones(1, 1, dtype=torch.bool,
                                                       device=radar.device),
                                      indices=torch.zeros(1, 1, dtype=torch.long,
                                                          device=radar.device),
                                      distances_m=torch.full((1, 1), 0.1,
                                                              device=radar.device)),
            attention=torch.ones(1, 1, 1, device=radar.device))


class ToyStudent(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(decoder_from_conditioned=True)
        self.shared = nn.Linear(1, 1, bias=False)
        self.shared.weight.data.fill_(0.2)

    def encode_conditioning(self, radar, valid):
        coords = torch.tensor([[0, 1, 32, 32]], device=radar.device)
        features = self.shared(radar[:, 0, 3:4])
        sites = SparseSites(coords, features, GRID.scale_shape(8), 8)
        return sites, sites, None

    def forward(self, radar, valid):
        feature, _, _ = self.encode_conditioning(radar, valid)
        coords = torch.tensor([[0, 8, 256, 256]], device=radar.device)
        logits = SparseSites(coords, feature.features, GRID.shape_zyx)
        return StudentOutput(feature, logits)


def test_quiz_map_gradient_updates_teacher_without_a_lidar_decoder_path(monkeypatch):
    from models.radar_lidar_crossmodal_v1 import meta_occupancy

    monkeypatch.setattr(meta_occupancy, "student_targets", lambda *args: SimpleNamespace(
        occupied=torch.tensor([True]), known_free=torch.tensor([False]),
        clean_point_count=torch.tensor([1])))
    teacher, student = ToyTeacher(), ToyStudent()
    optimizer = torch.optim.SGD(teacher.core.parameters(), lr=0.2)
    config = MetaOccupancyConfig(inner_lr=0.5, teacher_geometry_weight=0,
                                 teacher_fusion_weight=0)

    def batch(frame, radar_strength):
        radar = torch.zeros(1, 1, 7)
        radar[0, 0, 3] = radar_strength
        return {"frame_id": [frame], "radar": radar,
                "radar_valid": torch.ones(1, 1, dtype=torch.bool),
                "clean_lidar": torch.tensor([[[0.8, 0.0, 0.0, 1.0]]]),
                "clean_lidar_valid": torch.ones(1, 1, dtype=torch.bool)}

    paired, quiz = batch("A", 1.0), batch("B", 1.5)
    before_teacher = teacher.core.weight.detach().clone()
    before_student = student.shared.weight.detach().clone()
    prediction_before = student(quiz["radar"], quiz["radar_valid"]).confidence_logits.features.detach().clone()
    result = meta_teacher_step(teacher, student, paired, quiz, optimizer, config)
    assert not result.get("skipped", False)
    assert result["quiz"]["known"] == 1
    assert not torch.equal(teacher.core.weight, before_teacher)
    torch.testing.assert_close(student.shared.weight, before_student)
    prediction_after = student(quiz["radar"], quiz["radar_valid"]).confidence_logits.features
    torch.testing.assert_close(prediction_after, prediction_before)


def test_map_loss_reaches_actual_cross_attention_query_through_virtual_step():
    torch.manual_seed(11)
    core = BottleneckTeacherCore(TeacherConfig(radius_m=3.2))
    radar = SparseSites(torch.tensor([[0, 1, 32, 32]]), torch.randn(1, 64),
                        GRID.scale_shape(8), 8)
    lidar = SparseSites(torch.tensor([[0, 1, 32, 32], [0, 1, 32, 33]]),
                        torch.randn(2, 64), GRID.scale_shape(8), 8)
    indices = torch.full((1, 24), -1, dtype=torch.long)
    indices[0, :2] = torch.tensor([0, 1])
    distances = torch.full((1, 24), float("inf"))
    distances[0, :2] = torch.tensor([0.5, 1.5])
    target = core(radar, lidar, Neighborhood(indices, distances)).fused
    weight = nn.Parameter(torch.eye(64) * 0.1)
    student_features = radar.replace_features(radar.features @ weight.T)
    distill = feature_distillation(student_features, target, beta=10,
                                   detach_teacher=False)
    virtual = weight - 1.0 * torch.autograd.grad(distill, weight, create_graph=True)[0]
    radar_only_quiz_logit = (virtual[0] * radar.features[0].detach()).sum()
    map_loss = nn.functional.binary_cross_entropy_with_logits(
        radar_only_quiz_logit, torch.ones_like(radar_only_quiz_logit))
    query_gradient = torch.autograd.grad(map_loss, core.query.weight)[0]
    assert query_gradient.norm().item() > 0


def test_meta_cache_batch_offsets_and_cached_targets(tmp_path):
    def row(frame):
        return {
            "frame_id": frame, "radar_coords": torch.tensor([[0, 1, 2, 3]]),
            "radar_features": torch.ones(1, 64),
            "lidar_coords": torch.tensor([[0, 1, 2, 3]]),
            "lidar_features": torch.ones(1, 64),
            "neighbor_indices": torch.tensor([[0, -1]]),
            "neighbor_distances": torch.tensor([[0.2, float("inf")]]),
            "candidate_coords": torch.tensor([[0, 8, 16, 24]]),
            "observed_coords": torch.tensor([[0, 8, 16, 24]]),
            "occupied": torch.tensor([True]),
            "known_free": torch.tensor([False]),
            "support": torch.tensor([True]),
            "clean_supported_points": 2,
            "represented_clean_points_supported": 2,
        }
    manifest = {"frame_ids": ["1", "2"], "max_candidates": 2}
    cache = MetaOccupancyCache(tmp_path, manifest, create=True)
    for frame in ("1", "2"):
        cache.write(frame, {key: value for key, value in row(frame).items()
                            if key != "frame_id"})
    cache.check_complete()
    batch = prepare_meta_batch(collate_meta_cache([cache[0], cache[1]]), "cpu")
    assert batch["neighbors"].indices.tolist() == [[0, -1], [1, -1]]
    assert batch["candidate_coords"][:, 0].tolist() == [0, 1]
    assert batch["observed_coords"][:, 0].tolist() == [0, 1]
    assert batch["clean_supported_points"] == 4
    assert batch["targets"].represented_clean_points_supported == 4


def test_cached_virtual_student_step_reaches_teacher_parameter():
    class Core(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.tensor(0.8))

        def forward(self, radar, lidar, neighbors):
            fused = radar.replace_features(lidar.features * self.weight)
            return SimpleNamespace(
                fused=fused, lidar_frozen=lidar, neighbors=neighbors,
                attention=torch.ones(1, 1, 1))

    class Teacher(nn.Module):
        def __init__(self):
            super().__init__()
            self.core = Core()

    class Student(nn.Module):
        def __init__(self):
            super().__init__()
            self.config = SimpleNamespace(decoder_from_conditioned=True)
            self.shared = nn.Linear(1, 1, bias=False)
            self.shared.weight.data.fill_(0.2)

        def encode_conditioning_sites(self, radar):
            feature = radar.replace_features(self.shared(radar.features))
            return feature, feature

        def forward_encoded(self, radar):
            feature, _ = self.encode_conditioning_sites(radar)
            confidence = SparseSites(
                torch.tensor([[0, 8, 256, 256]]), feature.features,
                GRID.shape_zyx)
            return StudentOutput(feature, confidence)

    def batch(frame, strength):
        coarse = torch.tensor([[0, 1, 32, 32]])
        radar = SparseSites(coarse, torch.tensor([[strength]]), GRID.scale_shape(8), 8)
        lidar = SparseSites(coarse, torch.tensor([[1.0]]), GRID.scale_shape(8), 8)
        return {
            "frame_id": [frame], "radar_sites": radar, "lidar_sites": lidar,
            "neighbors": Neighborhood(torch.tensor([[0]]), torch.tensor([[0.1]])),
            "candidate_coords": torch.tensor([[0, 8, 256, 256]]),
            "targets": CachedOccupancyTargets(torch.tensor([True]),
                                              torch.tensor([False]), 1),
            "support": torch.tensor([True]),
        }

    teacher, student = Teacher(), Student()
    optimizer = torch.optim.SGD(teacher.core.parameters(), lr=0.2)
    before = teacher.core.weight.detach().clone()
    result = meta_teacher_step(
        teacher, student, batch("A", 1.0), batch("B", 1.5), optimizer,
        MetaOccupancyConfig(inner_lr=0.5, teacher_geometry_weight=0,
                            teacher_fusion_weight=0))
    assert not result.get("skipped", False)
    assert not torch.equal(teacher.core.weight, before)

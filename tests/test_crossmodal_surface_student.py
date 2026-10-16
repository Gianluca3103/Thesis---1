"""Focused checks for radar-only surface proposals and edge-only interpolation."""

from __future__ import annotations

import numpy as np
import torch
from types import SimpleNamespace

from models.radar_lidar_stage1.online_augmentation import GeometricAugmentationConfig
from models.radar_lidar_stage1.sparse import SparseSites
from models.radar_lidar_crossmodal_v1.edge_aware_surface import edge_aware_surface
from models.radar_lidar_crossmodal_v1.encoder import GRID
from models.radar_lidar_crossmodal_v1.surface_student import (
    RadarOnlySurfaceStudent, RadarSurfaceHead, SurfaceProposals, SurfaceStudentConfig,
    SurfaceStudentOutput,
)
from models.radar_lidar_crossmodal_v1.previous_encoder import PreviousStage1Sparse
from models.radar_lidar_crossmodal_v1.surface_student_loss import (
    clean_surface_sites, surface_student_objective,
)


def _sites(coords, channels, stride):
    shape = GRID.scale_shape(stride)
    coordinates = torch.tensor(coords, dtype=torch.long)
    features = torch.randn((len(coords), channels), dtype=torch.float32)
    return SparseSites(coordinates, features, shape, stride)


def _radar_levels():
    return (_sites([[0, 8, 256, 256], [0, 8, 256, 257]], 16, 1),
            _sites([[0, 4, 128, 128]], 32, 2),
            _sites([[0, 2, 64, 64]], 64, 4),
            _sites([[0, 1, 32, 32]], 64, 8))


def test_head_predicts_two_continuous_locations_without_filling_children():
    config = SurfaceStudentConfig()
    head = RadarSurfaceHead(config)
    result = head(_radar_levels())
    assert result.xyz.shape == (1, 2, 3)
    assert result.logits.shape == (1, 2)
    assert result.guide.shape == (1, 32)
    assert torch.all(torch.abs(result.xyz - result.anchor_xyz[:, None]) <= 1.5)
    assert result.anchor_batch.tolist() == [0]


def test_geometry_and_confidence_supervise_moving_proposals():
    torch.manual_seed(2)
    config = SurfaceStudentConfig()
    head = RadarSurfaceHead(config)
    levels = _radar_levels()
    proposals = head(levels)
    conditioned = levels[3]
    output = SurfaceStudentOutput(conditioned, proposals)
    teacher = conditioned.replace_features(torch.zeros_like(conditioned.features))
    clean_point = proposals.anchor_xyz.detach()[:, None, :]
    clean = clean_surface_sites(clean_point, torch.ones((1, 1), dtype=torch.bool))
    loss, pieces = surface_student_objective(output, teacher, clean, config)
    assert torch.isfinite(loss)
    assert pieces["anchors_with_targets"] == 1
    assert pieces["proposals"] == 2
    loss.backward()
    assert head.slots.weight.grad is not None
    assert torch.isfinite(head.slots.weight.grad).all()
    assert head.slots.weight.grad.abs().sum() > 0


def test_edge_field_links_compatible_seeds_and_keeps_batches_separate():
    xyz = np.array([[0.0, 0.0, 0.0], [.4, 0.0, 0.0],
                    [0.0, 0.0, 0.0], [.4, 0.0, 0.0]], dtype=np.float32)
    score = np.full(4, .8, dtype=np.float32)
    feature = np.tile(np.array([[1., 0.]], dtype=np.float32), (4, 1))
    groups = np.array([0, 1, 2, 3])
    batches = np.array([0, 0, 1, 1])
    field = edge_aware_surface(xyz, score, feature, groups, batches)
    assert field.accepted_edges == 2
    assert set(field.batch_index.tolist()) == {0, 1}
    assert np.array_equal(field.coordinates_xyz[field.batch_index == 0],
                          field.coordinates_xyz[field.batch_index == 1])
    assert np.all(np.abs(field.centers_xyz()[:, 0]) < 1.)


def test_edge_field_does_not_bridge_unsupported_gaps():
    xyz = np.array([[0., 0., 0.], [4., 0., 0.]], dtype=np.float32)
    field = edge_aware_surface(xyz, np.array([.9, .9]),
                               np.ones((2, 2)), np.array([0, 1]))
    assert field.accepted_edges == 0
    centers = field.centers_xyz()
    assert not np.any((centers[:, 0] > 1.) & (centers[:, 0] < 3.))


def test_clean_targets_do_not_cross_sample_boundaries():
    config = SurfaceStudentConfig()
    conditioned = _sites([[0, 1, 32, 32]], 64, 8)
    proposals = SurfaceProposals(
        xyz=torch.zeros((1, 2, 3), requires_grad=True),
        logits=torch.zeros((1, 2), requires_grad=True),
        anchor_xyz=torch.zeros((1, 3)), anchor_batch=torch.tensor([0]),
        guide=torch.ones((1, 32)))
    clean = _sites([[1, 8, 256, 256]], 1, 1)
    _, pieces = surface_student_objective(
        SurfaceStudentOutput(conditioned, proposals), conditioned, clean, config)
    assert pieces["anchors_with_targets"] == 0
    assert pieces["geometry"].item() == 0


def test_view_contains_only_raw_and_edge_confidence_panels(tmp_path):
    from scripts.compare_crossmodal_surface import _write_html

    path = tmp_path / "view.html"
    raw = np.array([[0., 0., 0.], [.4, 0., 0.]])
    _write_html(path, "00405", 1, raw, raw, raw, np.array([.7, .8]),
                raw, np.array([.6, .9]), threshold=.25, links=1)
    page = path.read_text(encoding="utf-8")
    assert "Raw predicted surfaces" in page
    assert "Edge-aware interpolation" in page
    assert "Ellipsoid" not in page
    assert "Trilinear" not in page
    assert '"frame":"00405"' in page


def test_end_to_end_student_inference_needs_only_radar(tmp_path):
    checkpoint = tmp_path / "radar_encoder.pth"
    torch.save(PreviousStage1Sparse(modality="radar").state_dict(), checkpoint)
    student = RadarOnlySurfaceStudent(str(checkpoint)).eval()
    radar = torch.tensor([[[10., 0., 0., 1., 0., 0., 0.],
                           [10.2, .1, 0., 1., 0., 0., 0.]]])
    result = student(radar, torch.ones((1, 2), dtype=torch.bool))
    assert result.surface.xyz.shape[1:] == (2, 3)
    assert result.surface.xyz.shape[0] > 0
    assert result.conditioning.features.shape[1] == 64
    assert "surface_xyz" in result.exported()
    assert "confidence_coords" not in result.exported()


def test_surface_checkpoint_reloads_for_raw_vs_edge_view(tmp_path):
    from models.radar_lidar_crossmodal_v1.train_surface_student import (
        _save, load_surface_student,
    )

    encoder = tmp_path / "radar_encoder.pth"
    torch.save(PreviousStage1Sparse(modality="radar").state_dict(), encoder)
    student = RadarOnlySurfaceStudent(str(encoder)).eval()
    optimizer = torch.optim.AdamW((p for p in student.parameters() if p.requires_grad))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=1)
    checkpoint = tmp_path / "surface.ckpt"
    args = SimpleNamespace(radar_variant="radar_20frames_verified_doppler_radial",
                           augmentation=GeometricAugmentationConfig())
    _save(checkpoint, student, optimizer, scheduler, tmp_path / "teacher.ckpt",
          "mock-hash", 3, args, {"val": {"surface_f1_20cm": .2}})
    loaded, saved = load_surface_student(checkpoint, "cpu")
    assert saved["epoch"] == 3
    assert loaded.config == student.config
    assert all(torch.equal(value, loaded.state_dict()[key])
               for key, value in student.state_dict().items())

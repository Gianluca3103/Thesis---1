"""Focused checks for radar-only surface proposals and edge-only interpolation."""

from __future__ import annotations

import numpy as np
import torch
from dataclasses import asdict
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
from models.radar_lidar_crossmodal_v1.train_teacher import _hash
from models.radar_lidar_crossmodal_v1.surface_student_loss import (
    surface_student_objective,
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


def test_head_uses_every_fine_radar_site_without_filling_children():
    config = SurfaceStudentConfig()
    head = RadarSurfaceHead(config)
    levels = _radar_levels()
    result = head(levels)
    assert result.xyz.shape == (2, 24, 3)
    assert result.logits.shape == (2, 24)
    assert result.guide.shape == (2, 32)
    assert torch.equal(result.anchor_xyz, levels[0].centers_xyz(GRID))
    assert torch.all(torch.abs(result.xyz - result.anchor_xyz[:, None]) <= 1.5)
    assert result.anchor_batch.tolist() == [0, 0]


def test_previous_stride_eight_checkpoint_geometry_remains_supported():
    config = SurfaceStudentConfig(anchor_stride=8, proposals_per_anchor=2)
    result = RadarSurfaceHead(config)(_radar_levels())
    assert result.xyz.shape == (1, 2, 3)
    assert result.anchor_batch.tolist() == [0]


def test_fine_anchors_do_not_cross_sample_boundaries():
    fine = _sites([[0, 8, 256, 256], [1, 8, 256, 256]], 16, 1)
    levels = (fine, _sites([[0, 4, 128, 128], [1, 4, 128, 128]], 32, 2),
              _sites([[0, 2, 64, 64], [1, 2, 64, 64]], 64, 4),
              _sites([[0, 1, 32, 32], [1, 1, 32, 32]], 64, 8))
    result = RadarSurfaceHead(SurfaceStudentConfig())(levels)
    assert result.xyz.shape == (2, 24, 3)
    assert result.anchor_batch.tolist() == [0, 1]


def test_sequential_head_queries_own_distilled_parent_and_keeps_batches_separate():
    fine = _sites([[0, 8, 256, 256], [1, 8, 256, 256]], 16, 1)
    levels = (fine, _sites([[0, 4, 128, 128], [1, 4, 128, 128]], 32, 2),
              _sites([[0, 2, 64, 64], [1, 2, 64, 64]], 64, 4),
              _sites([[0, 1, 32, 32], [1, 1, 32, 32]], 64, 8))
    levels[3].features[0].fill_(1.)
    levels[3].features[1].fill_(2.)
    head = RadarSurfaceHead(SurfaceStudentConfig(decoder_from_conditioned=True))
    captured = []
    handle = head.query.register_forward_pre_hook(lambda _module, args: captured.append(args[0]))
    try:
        result = head(levels)
    finally:
        handle.remove()
    assert result.anchor_batch.tolist() == [0, 1]
    assert torch.equal(captured[0], levels[3].features)


def test_sequential_head_rejects_missing_distilled_parent():
    levels = list(_radar_levels())
    levels[3] = _sites([[0, 1, 32, 35]], 64, 8)
    head = RadarSurfaceHead(SurfaceStudentConfig(decoder_from_conditioned=True))
    try:
        head(tuple(levels))
    except RuntimeError as exc:
        assert "no conditioned stride-8 parent" in str(exc)
    else:
        raise AssertionError("Missing distilled parent was silently accepted")


def test_occupancy_and_hit_reward_supervise_moving_proposals():
    torch.manual_seed(2)
    config = SurfaceStudentConfig()
    head = RadarSurfaceHead(config)
    levels = _radar_levels()
    proposals = head(levels)
    conditioned = levels[3]
    output = SurfaceStudentOutput(conditioned, proposals)
    clean_point = proposals.anchor_xyz.detach()[None, :, :]
    loss, pieces, field, targets = surface_student_objective(
        output, clean_point, torch.ones((1, 2), dtype=torch.bool), config)
    assert torch.isfinite(loss)
    assert pieces["pointer_slots"] == 48
    assert pieces["occupied_sites"] > 0
    assert len(field.coords) == len(targets.occupied)
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
    clean_point = torch.tensor([[[0., 0., 0.]], [[.1, .1, .1]]])
    valid = torch.tensor([[False], [True]])
    _, pieces, _, _ = surface_student_objective(
        SurfaceStudentOutput(conditioned, proposals), clean_point, valid, config)
    assert pieces["known_sites"] == 0
    assert pieces["hit_reward"].item() == 0


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


def test_view_can_show_fixed_threshold_voxel_error_panels(tmp_path):
    from scripts.compare_crossmodal_surface import _write_html

    path = tmp_path / "audit.html"
    points = np.array([[0., 0., 0.]])
    _write_html(path, "00405", 1, points, points, points, np.array([.7]),
                points, np.array([.7]), threshold=.6, links=0,
                audit_clouds=[{"name": "Raw audit", "xyz": points.tolist(),
                               "color": ["#ff6868"], "summary": "TP 0 · FP 1 · FN 1"}])
    page = path.read_text(encoding="utf-8")
    assert "Raw audit" in page
    assert "fixed export threshold" in page
    assert "#ff6868" in page


def test_end_to_end_student_inference_needs_only_radar(tmp_path):
    checkpoint = tmp_path / "radar_encoder.pth"
    torch.save(PreviousStage1Sparse(modality="radar").state_dict(), checkpoint)
    student = RadarOnlySurfaceStudent(str(checkpoint)).eval()
    radar = torch.tensor([[[10., 0., 0., 1., 0., 0., 0.],
                           [10.2, .1, 0., 1., 0., 0., 0.]]])
    result = student(radar, torch.ones((1, 2), dtype=torch.bool))
    assert result.surface.xyz.shape[1:] == (24, 3)
    assert result.surface.xyz.shape[0] > 0
    assert result.conditioning.features.shape[1] == 64
    assert "surface_xyz" in result.exported()
    assert "confidence_coords" in result.exported()


def test_trained_student_is_frozen_before_radar_only_surface_decoding(tmp_path):
    checkpoint = tmp_path / "radar_encoder.pth"
    torch.save(PreviousStage1Sparse(modality="radar").state_dict(), checkpoint)
    config = SurfaceStudentConfig(decoder_from_conditioned=True,
                                  distillation_weight=0)
    model = RadarOnlySurfaceStudent(str(checkpoint), config)
    source = {"version": "crossmodal_feature_student_v1",
              "config": {"feature_dim": 64},
              "radar_checkpoint": str(checkpoint.resolve()),
              "radar_sha256": _hash(checkpoint),
              "model": {key: value.clone() for key, value in model.state_dict().items()
                        if key.startswith(("radar_encoder.", "shared.",
                                           "feature_head.", "feature_norm."))}}
    source["model"]["feature_norm.bias"].fill_(0.25)
    model.load_feature_student(source)
    assert all(not parameter.requires_grad for name, parameter in model.named_parameters()
               if name.startswith(("radar_encoder.", "shared.", "feature_head.",
                                   "feature_norm.")))
    assert all(parameter.requires_grad for name, parameter in model.surface_head.named_parameters()
               if not name.startswith("guide."))
    radar = torch.tensor([[[10., 0., 0., 1., 0., 0., 0.],
                           [10.2, .1, 0., 1., 0., 0., 0.]]])
    result = model(radar, torch.ones((1, 2), dtype=torch.bool))
    assert result.surface.xyz.shape[1:] == (24, 3)
    loss, pieces, _, _ = surface_student_objective(
        result, radar[..., :3], torch.ones((1, 2), dtype=torch.bool), config)
    assert "distillation" not in pieces
    loss.backward()
    assert any(parameter.grad is not None for parameter in model.surface_head.parameters())
    assert all(parameter.grad is None for parameter in model.shared.parameters())


def test_decoder_epoch_uses_radar_and_clean_labels_without_teacher(tmp_path):
    from models.radar_lidar_crossmodal_v1.train_surface_student import _epoch

    checkpoint = tmp_path / "radar_encoder.pth"
    torch.save(PreviousStage1Sparse(modality="radar").state_dict(), checkpoint)
    config = SurfaceStudentConfig(decoder_from_conditioned=True,
                                  distillation_weight=0)
    model = RadarOnlySurfaceStudent(str(checkpoint), config)
    model.load_feature_student({
        "version": "crossmodal_feature_student_v1",
        "config": {"feature_dim": 64}, "radar_checkpoint": str(checkpoint.resolve()),
        "radar_sha256": _hash(checkpoint),
        "model": {key: value.clone() for key, value in model.state_dict().items()}})
    points = torch.tensor([[[10., 0., 0., 1., 0., 0., 0.],
                            [10.2, .1, 0., 1., 0., 0., 0.]]])
    clean = torch.tensor([[[10., 0., 0., 1.], [10.2, .1, 0., 1.]]])
    batch = {"frame_id": ["synthetic"], "radar": points,
             "radar_valid": torch.ones((1, 2), dtype=torch.bool),
             "clean_lidar": clean,
             "clean_lidar_valid": torch.ones((1, 2), dtype=torch.bool)}
    dataset = SimpleNamespace(set_epoch=lambda epoch: None)
    optimizer = torch.optim.AdamW(model.surface_head.parameters(), lr=1e-4)
    result = _epoch(model, dataset, [batch], "cpu", optimizer=optimizer)
    assert "loss_distillation" not in result
    assert "loss_bce" in result and "loss_dice" in result
    assert result["pointer_slots"] > 0


def test_surface_checkpoint_reloads_for_raw_vs_edge_view(tmp_path):
    from models.radar_lidar_crossmodal_v1.train_surface_student import (
        _save, load_surface_student,
    )

    encoder = tmp_path / "radar_encoder.pth"
    torch.save(PreviousStage1Sparse(modality="radar").state_dict(), encoder)
    args = SimpleNamespace(radar_variant="radar_20frames_verified_doppler_radial",
                           augmentation=GeometricAugmentationConfig())
    configs = (SurfaceStudentConfig(),
               SurfaceStudentConfig(anchor_stride=8, proposals_per_anchor=2))
    for index, config in enumerate(configs):
        student = RadarOnlySurfaceStudent(str(encoder), config).eval()
        optimizer = torch.optim.AdamW((p for p in student.parameters() if p.requires_grad))
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=1)
        checkpoint = tmp_path / f"surface_{index}.ckpt"
        _save(checkpoint, student, optimizer, scheduler, tmp_path / "teacher.ckpt",
              "mock-hash", 3, args, {"val": {"proposal_utility": .2}})
        loaded, saved = load_surface_student(checkpoint, "cpu")
        assert saved["epoch"] == 3
        assert loaded.config == student.config
        assert all(torch.equal(value, loaded.state_dict()[key])
                   for key, value in student.state_dict().items())
        if index == 1:
            legacy = torch.load(checkpoint, map_location="cpu", weights_only=False)
            legacy["version"] = "crossmodal_surface_student_v1"
            legacy["config"].pop("decoder_from_conditioned")
            torch.save(legacy, checkpoint)
            old_model, _ = load_surface_student(checkpoint, "cpu")
            assert old_model.config.decoder_from_conditioned is False
            radar = torch.tensor([[[10., 0., 0., 1., 0., 0., 0.]]])
            legacy_output = old_model(radar, torch.ones((1, 1), dtype=torch.bool))
            assert "confidence_coords" not in legacy_output.exported()


def test_new_pointer_head_can_warm_start_four_trained_slots(tmp_path):
    from models.radar_lidar_crossmodal_v1.train_surface_student import (
        _initialize_head_from_surface,
    )

    encoder = tmp_path / "radar_encoder.pth"
    torch.save(PreviousStage1Sparse(modality="radar").state_dict(), encoder)
    old_config = SurfaceStudentConfig(proposals_per_anchor=4,
                                      decoder_from_conditioned=True,
                                      distillation_weight=0)
    old = RadarOnlySurfaceStudent(str(encoder), old_config)
    with torch.no_grad():
        old.surface_head.slots.weight.fill_(.123)
        old.surface_head.slots.bias.fill_(.234)
    path = tmp_path / "old_surface.ckpt"
    torch.save({"version": "crossmodal_surface_student_v3_feature_first",
                "config": asdict(old_config), "model": old.state_dict(),
                "feature_student_sha256": "matching-feature-student"}, path)
    new = RadarOnlySurfaceStudent(
        str(encoder), SurfaceStudentConfig(decoder_from_conditioned=True,
                                          distillation_weight=0))
    _initialize_head_from_surface(new, path, "matching-feature-student")
    assert torch.allclose(new.surface_head.slots.weight.view(24, 4, -1)[:4],
                          old.surface_head.slots.weight.view(4, 4, -1))
    assert torch.allclose(new.surface_head.slots.bias.view(24, 4)[:4],
                          old.surface_head.slots.bias.view(4, 4))
    assert not torch.allclose(new.surface_head.slots.weight.view(24, 4, -1)[4:],
                              torch.full_like(new.surface_head.slots.weight.view(24, 4, -1)[4:], .123))

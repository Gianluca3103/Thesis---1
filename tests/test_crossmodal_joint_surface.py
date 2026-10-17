"""Focused checks for the frozen-teacher joint fine-tuning ablation."""

from __future__ import annotations

from types import SimpleNamespace
import json

import pytest
import torch

from scripts.compare_joint_surface_ablation import compare

from models.radar_lidar_crossmodal_v1.feature_student import FeatureStudentConfig
from models.radar_lidar_crossmodal_v1.previous_encoder import PreviousStage1Sparse
from models.radar_lidar_crossmodal_v1.surface_student import (
    RadarOnlySurfaceStudent, SurfaceStudentConfig,
)
from models.radar_lidar_crossmodal_v1.surface_student_loss import surface_student_objective
from models.radar_lidar_crossmodal_v1.train_joint_surface_student import (
    _epoch, _save, configure_trainable, joint_losses,
)
from models.radar_lidar_crossmodal_v1.train_surface_student import (
    JOINT_VERSION, load_surface_student,
)


def _setup(tmp_path):
    encoder = tmp_path / "radar_encoder.pth"
    torch.save(PreviousStage1Sparse(modality="radar").state_dict(), encoder)
    config = SurfaceStudentConfig(decoder_from_conditioned=True,
                                  pointer_voxel_mode=True)
    model = RadarOnlySurfaceStudent(str(encoder), config)
    with torch.no_grad():
        model.surface_head.slots.weight.mul_(0.001)
        model.surface_head.slots.bias.zero_()
    radar = torch.tensor([[[10., 0., 0., 1., 0., 0., 0.],
                           [10.2, .1, 0., 1., 0., 0., 0.]]])
    batch = {"frame_id": ["00405"], "radar": radar,
             "radar_valid": torch.ones((1, 2), dtype=torch.bool),
             "clean_lidar": radar[..., :4].clone(),
             "clean_lidar_valid": torch.ones((1, 2), dtype=torch.bool)}
    return model, batch


def _teacher_sites(output):
    target = output.conditioning.features.detach().clone().add(0.25)
    return output.conditioning.replace_features(target)


def test_joint_pointer_and_distillation_both_reach_feature_student(tmp_path):
    model, batch = _setup(tmp_path)
    configure_trainable(model, freeze_feature_student=False)
    assert all(not p.requires_grad for p in model.radar_encoder.parameters())
    output = model(batch["radar"], batch["radar_valid"])
    teacher = _teacher_sites(output)
    total, pointer, distillation, pieces, _, _ = joint_losses(
        output, teacher, batch["clean_lidar"], batch["clean_lidar_valid"],
        model.config, FeatureStudentConfig(), distillation_weight=1,
        pointer_weight=1)
    assert pieces["occupied_sites"] > 0
    assert torch.allclose(total, pointer + distillation)
    pointer.backward(retain_graph=True)
    assert any(p.grad is not None and p.grad.abs().sum() > 0
               for p in model.shared.parameters())
    model.zero_grad(set_to_none=True)
    distillation.backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0
               for p in model.feature_head.parameters())
    assert all(p.grad is None for p in model.radar_encoder.parameters())
    assert teacher.features.grad is None


def test_head_only_control_reproduces_pointer_loss(tmp_path):
    model, batch = _setup(tmp_path)
    configure_trainable(model, freeze_feature_student=True)
    output = model(batch["radar"], batch["radar_valid"])
    expected, _, _, _ = surface_student_objective(
        output, batch["clean_lidar"], batch["clean_lidar_valid"], model.config)
    total, pointer, distillation, _, _, _ = joint_losses(
        output, _teacher_sites(output), batch["clean_lidar"],
        batch["clean_lidar_valid"], model.config, FeatureStudentConfig(),
        distillation_weight=0, pointer_weight=1)
    assert torch.allclose(total, expected)
    assert torch.allclose(pointer, expected)
    assert distillation > 0
    total.backward()
    assert any(p.grad is not None for p in model.surface_head.parameters())
    assert all(p.grad is None for p in model.shared.parameters())


def test_online_teacher_is_only_a_frozen_training_target(tmp_path):
    model, batch = _setup(tmp_path)
    configure_trainable(model, freeze_feature_student=False)

    class Teacher(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.bias = torch.nn.Parameter(torch.ones(64))

        def forward(self, radar, radar_valid, clean, clean_valid):
            sites = model.radar_encoder(radar, radar_valid)
            return SimpleNamespace(fused=sites.replace_features(
                sites.features + self.bias))

    teacher = Teacher()
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),
                                  lr=1e-4)
    report = _epoch(model, teacher, None, FeatureStudentConfig(),
                    SimpleNamespace(set_epoch=lambda epoch: None), [batch], "cpu",
                    distillation_weight=1, pointer_weight=1, optimizer=optimizer)
    assert report["loss_distillation"] > 0
    assert teacher.bias.grad is None


@pytest.mark.parametrize("freeze", [False, True])
def test_joint_epoch_reads_teacher_cache_and_reports_matching_metrics(tmp_path, freeze):
    model, batch = _setup(tmp_path)
    configure_trainable(model, freeze_feature_student=freeze)
    with torch.no_grad():
        output = model(batch["radar"], batch["radar_valid"])
    entry = {"bottleneck_coords": output.conditioning.coords.cpu(),
             "teacher_features": output.conditioning.features.detach().cpu() + .25}
    cache = SimpleNamespace(_read=lambda frame_id: entry)
    dataset = SimpleNamespace(set_epoch=lambda epoch: None)
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),
                                  lr=1e-4)
    report = _epoch(model, None, cache, FeatureStudentConfig(), dataset,
                    [batch], "cpu", distillation_weight=0 if freeze else 1,
                    pointer_weight=1, optimizer=optimizer)
    assert report["frames"] == 1
    assert report["feature_elements"] > 0
    assert report["loss_distillation"] > 0
    assert report["occupied_sites"] > 0
    assert report["loss_total"] == pytest.approx(
        report["loss_pointer"] + (0 if freeze else report["loss_distillation"]))
    validation = _epoch(model, None, cache, FeatureStudentConfig(), dataset,
                        [batch], "cpu", distillation_weight=0 if freeze else 1,
                        pointer_weight=1)
    assert "proposal_utility" in validation
    assert "verified_proposal_precision" in validation


def test_joint_checkpoint_reloads_for_radar_only_inference(tmp_path):
    model, batch = _setup(tmp_path)
    configure_trainable(model, freeze_feature_student=False)
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=1)
    args = SimpleNamespace(surface_checkpoint=tmp_path / "initial.ckpt",
                           surface_sha256="surface-hash", feature_sha256="feature-hash",
                           teacher_checkpoint=tmp_path / "teacher.ckpt",
                           teacher_sha256="teacher-hash", freeze_feature_student=False,
                           distillation_weight=1., pointer_weight=1., seed=42)
    checkpoint = tmp_path / "joint.ckpt"
    _save(checkpoint, model, optimizer, scheduler, 1, args,
          {"radar_variant": "radar_20frames_verified_doppler_radial"},
          {"val": {"proposal_utility": 1.0}})
    reloaded, saved = load_surface_student(checkpoint, "cpu")
    assert saved["version"] == JOINT_VERSION
    assert reloaded.config.pointer_voxel_mode
    with torch.no_grad():
        result = reloaded(batch["radar"], batch["radar_valid"])
    assert "confidence_coords" in result.exported()
    assert result.conditioning.features.shape[1] == 64


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_joint_cuda_one_batch_smoke(tmp_path):
    model, batch = _setup(tmp_path)
    model = model.cuda()
    configure_trainable(model, freeze_feature_student=False)
    with torch.no_grad():
        output = model(batch["radar"].cuda(), batch["radar_valid"].cuda())
    entry = {"bottleneck_coords": output.conditioning.coords.cpu(),
             "teacher_features": output.conditioning.features.detach().cpu() + .25}
    cache = SimpleNamespace(_read=lambda frame_id: entry)
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),
                                  lr=1e-4)
    report = _epoch(model, None, cache, FeatureStudentConfig(),
                    SimpleNamespace(set_epoch=lambda epoch: None), [batch], "cuda",
                    distillation_weight=1, pointer_weight=1, optimizer=optimizer)
    assert torch.isfinite(torch.tensor(report["loss_total"]))
    assert report["occupied_sites"] > 0


def test_ablation_report_compares_matched_full_validation(tmp_path):
    roots = [tmp_path / "control", tmp_path / "joint"]
    values = [(10., 20, .5), (12., 24, .5)]
    for root, arm, (utility, hits, precision) in zip(
            roots, ("control", "joint"), values):
        root.mkdir()
        validation = {"frames": 2, "proposal_utility": utility,
                      "distinct_correct_voxels": hits,
                      "verified_proposal_precision": precision}
        (root / "metrics.jsonl").write_text(
            json.dumps({"epoch": 5, "arm": arm,
                        "validation_scope": "full", "val": validation}) + "\n")
        torch.save({"initial_surface_sha256": "same-surface",
                    "initial_feature_sha256": "same-feature",
                    "teacher_sha256": "same-teacher", "seed": 42,
                    "distillation_weight": 0 if arm == "control" else 1,
                    "freeze_feature_student": arm == "control"},
                   root / "last.ckpt")
    report = compare(*roots)
    assert report["matched_final_epoch"] == 5
    assert report["matched_final_joint_minus_control"]["proposal_utility"] == 2
    assert report["matched_final_joint_minus_control"]["distinct_correct_voxels"] == 4

"""Feature distillation and the handoff to a separately trained surface decoder."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch.nn import functional as F

from models.radar_lidar_stage1.online_augmentation import GeometricAugmentationConfig
from models.radar_lidar_stage1.sparse import SparseSites
from models.radar_lidar_crossmodal_v1.encoder import GRID
from models.radar_lidar_crossmodal_v1.feature_student import (
    FeatureStudentConfig, RadarOnlyFeatureStudent, feature_distillation_loss)
from models.radar_lidar_crossmodal_v1.previous_encoder import PreviousStage1Sparse
from models.radar_lidar_crossmodal_v1.surface_student import (
    RadarOnlySurfaceStudent, SurfaceStudentConfig)
from models.radar_lidar_crossmodal_v1.train_feature_student import (
    _epoch, _save, cached_teacher_sites, load_feature_student)


def _encoder(tmp_path):
    path = tmp_path / "radar_encoder.pth"
    torch.save(PreviousStage1Sparse(modality="radar").state_dict(), path)
    return path


def _radar_batch():
    return {"frame_id": ["00405"],
            "radar": torch.tensor([[[10., 0., 0., 1., 0., 0., 0.],
                                     [10.2, .1, 0., 1., 0., 0., 0.]]]),
            "radar_valid": torch.ones((1, 2), dtype=torch.bool)}


def test_feature_student_requires_radar_only_and_has_no_confidence_decoder(tmp_path):
    model = RadarOnlyFeatureStudent(str(_encoder(tmp_path))).train()
    batch = _radar_batch()
    output = model(batch["radar"], batch["radar_valid"])
    assert output.stride == 8 and output.features.shape[1] == 64
    assert not hasattr(model, "confidence_decoder")
    assert all(not p.requires_grad for p in model.radar_encoder.parameters())


def test_distillation_updates_student_features_but_not_teacher_or_encoder(tmp_path):
    model = RadarOnlyFeatureStudent(str(_encoder(tmp_path)))
    batch = _radar_batch()
    output = model(batch["radar"], batch["radar_valid"])
    teacher_values = torch.randn_like(output.features, requires_grad=True)
    teacher = output.replace_features(teacher_values)
    loss = feature_distillation_loss(output, teacher, model.config)
    assert torch.allclose(loss, F.smooth_l1_loss(output.features, teacher_values,
                                               beta=1.0))
    loss.backward()
    assert teacher_values.grad is None
    assert any(p.grad is not None for p in model.shared.parameters())
    assert all(p.grad is None for p in model.radar_encoder.parameters())
    assert torch.isfinite(loss)


def test_distillation_rejects_wrong_bottleneck_coordinates():
    shape = GRID.scale_shape(8)
    one = SparseSites(torch.tensor([[0, 1, 1, 1]]), torch.ones((1, 64)), shape, 8)
    other = SparseSites(torch.tensor([[0, 1, 1, 2]]), torch.ones((1, 64)), shape, 8)
    with pytest.raises(ValueError, match="sites differ"):
        feature_distillation_loss(one, other, FeatureStudentConfig())


def test_cached_teacher_features_preserve_batch_boundaries():
    shape = GRID.scale_shape(8)
    coords = torch.tensor([[0, 1, 1, 1], [1, 1, 1, 1]])
    student = SparseSites(coords, torch.zeros((2, 64)), shape, 8)
    cache = SimpleNamespace(_read=lambda frame: {
        "bottleneck_coords": torch.tensor([[0, 1, 1, 1]]),
        "teacher_features": torch.full((1, 64), 1. if frame == "a" else 2.)})
    teacher = cached_teacher_sites(cache, ["a", "b"], student, "cpu")
    assert torch.equal(teacher.coords, coords)
    assert teacher.features[0, 0] == 1 and teacher.features[1, 0] == 2
    wrong = SparseSites._from_validated(torch.tensor([[0, 1, 1, 2], [1, 1, 1, 1]]),
                                        torch.zeros((2, 64)), shape, 8)
    with pytest.raises(ValueError, match="coordinates differ"):
        cached_teacher_sites(cache, ["a", "b"], wrong, "cpu")


def test_feature_epoch_uses_cached_teacher_and_no_lidar(tmp_path):
    model = RadarOnlyFeatureStudent(str(_encoder(tmp_path)))
    batch = _radar_batch()
    output = model(batch["radar"], batch["radar_valid"])
    cache = SimpleNamespace(_read=lambda frame: {
        "bottleneck_coords": output.coords.cpu(),
        "teacher_features": torch.randn_like(output.features).detach().cpu()})
    result = _epoch(model, None, SimpleNamespace(set_epoch=lambda epoch: None),
                    [batch], "cpu", cache=cache,
                    optimizer=torch.optim.AdamW((p for p in model.parameters()
                                                 if p.requires_grad), lr=1e-4))
    assert result["frames"] == 1 and result["feature_elements"] > 0
    assert result["loss_total"] == result["loss_distillation"]


def test_feature_checkpoint_handoff_freezes_representation(tmp_path):
    model = RadarOnlyFeatureStudent(str(_encoder(tmp_path)))
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=1)
    args = SimpleNamespace(radar_variant="radar_20frames_verified_doppler_radial",
                           augmentation=GeometricAugmentationConfig())
    checkpoint = tmp_path / "feature.ckpt"
    _save(checkpoint, model, optimizer, scheduler, tmp_path / "teacher.ckpt",
          "teacher-hash", 1, args, {"val": {"loss_distillation": .1}})
    reloaded, saved = load_feature_student(checkpoint, "cpu")
    assert saved["version"] == "crossmodal_feature_student_v1"
    surface = RadarOnlySurfaceStudent(saved["radar_checkpoint"],
                                      SurfaceStudentConfig(decoder_from_conditioned=True,
                                                           distillation_weight=0))
    surface.load_feature_student(saved)
    assert all(not p.requires_grad for name, p in surface.named_parameters()
               if name.startswith(("radar_encoder.", "shared.", "feature_head.",
                                   "feature_norm.")))
    assert all(p.requires_grad for name, p in surface.surface_head.named_parameters()
               if not name.startswith("guide."))
    batch = _radar_batch()
    assert torch.allclose(reloaded(batch["radar"], batch["radar_valid"]).features,
                          surface(batch["radar"], batch["radar_valid"]).conditioning.features)

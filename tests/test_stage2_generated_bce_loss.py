"""Standard final-decoder BCE for the separate conditional Stage-II trial."""

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from models.radar_lidar_stage2.clean_denoising import batched_hierarchy
from models.radar_lidar_stage2.generated_bce_loss import generated_occupancy_bce
from models.radar_lidar_stage2.sparse_residual import (
    LATENT_CHANNELS, SparseResidualStage2)
from scripts import evaluate_stage2_conditional_reconstruction as evaluator
from scripts import train_stage2_clean_denoiser as trainer
from scripts.train_stage2_conditional_from_scratch import mse_epoch_log


def test_standard_bce_uses_only_unobserved_gate_cells_and_has_normal_gradients():
    logits = torch.tensor([-2.0, 2.0, 3.0], requires_grad=True)
    target = torch.tensor([True, False, True])
    allowed = torch.tensor([True, True, False])
    loss, counts = generated_occupancy_bce(logits, target, allowed)
    expected = F.binary_cross_entropy_with_logits(
        logits[:2], target[:2].float())
    assert loss.item() == pytest.approx(expected.item())
    assert counts == {"positive_count": 1, "negative_count": 1}
    loss.backward()
    assert logits.grad[0] < 0
    assert logits.grad[1] > 0
    assert logits.grad[2] == 0


def test_empty_addition_gate_has_finite_zero_bce():
    logits = torch.tensor([1.0], requires_grad=True)
    loss, counts = generated_occupancy_bce(
        logits, torch.tensor([True]), torch.tensor([False]))
    assert counts == {"positive_count": 0, "negative_count": 0}
    assert loss.item() == 0
    loss.backward()
    assert logits.grad.item() == 0


def test_generated_bce_updates_denoiser_through_frozen_decoder(monkeypatch):
    class Diffusion(nn.Module):
        def __init__(self):
            super().__init__()
            self.scale = nn.Parameter(torch.tensor(1.0))

        def forward(self, noisy, faulty, radar, probability, present,
                    timestep, coords):
            return self.scale * noisy + 0.05 * radar

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.diffusion = Diffusion()
            self.radar_adapter = nn.Linear(2, LATENT_CHANNELS, bias=False)
            self.lpcr_scale = nn.Parameter(torch.tensor(0.7),
                                           requires_grad=False)

        def lidar_encoder(self, clean, coarse, point_ids):
            return torch.full((len(coarse), LATENT_CHANNELS), 0.25), torch.ones(
                len(coarse), dtype=torch.bool)

        def lpcr(self, latent, hierarchy, site_features):
            return self.lpcr_scale * latent[0, 0] + torch.tensor([0.0, -0.5]), None

    fine = torch.tensor([[0, 0, 0, 0], [0, 0, 0, 1]])
    coarse = fine[:1]
    monkeypatch.setattr(trainer, "_batch", lambda rows, device: (
        (fine, fine, fine, fine, coarse), torch.zeros((1, 3)),
        torch.zeros(1, dtype=torch.long)))
    monkeypatch.setattr(trainer, "_conditioned_inputs", lambda model, *args, **kwargs: (
        torch.ones((1, LATENT_CHANNELS)), torch.ones(1, dtype=torch.bool),
        model.radar_adapter(torch.ones((1, 2)))))
    monkeypatch.setattr(trainer, "offset_targets", lambda *args: (
        torch.tensor([True, False]), None, None))
    monkeypatch.setattr(trainer, "addition_context", lambda *args: type(
        "Context", (), {"observed": torch.tensor([False, False])})())
    model = Model()
    optimizer = torch.optim.SGD(
        [model.diffusion.scale, *model.radar_adapter.parameters()], lr=0)
    row = {"frame_id": "dummy", "faulty_xyz": torch.zeros((0, 3))}
    report = trainer._run_epoch(
        model, [[row]], optimizer, torch.linspace(0.999, 0.001, 1000),
        device="cpu", epoch=1, eval_timestep=499, seed=3103,
        condition_faulty_radar=True, train_radar_adapter=True,
        decoder_occupancy=trainer.GeneratedBCEConfig(
            rollout_steps=4, gradient_steps=2, loss_weight=0.1))
    assert report["decoder_positive_sites"] == 1
    assert report["decoder_negative_sites"] == 1
    assert report["total_loss"] == pytest.approx(
        report["noise_mse"] + 0.1 * report["decoder_occupancy_loss"])
    assert torch.isfinite(model.diffusion.scale.grad)
    assert model.radar_adapter.weight.grad is not None
    assert model.lpcr_scale.grad is None


def test_real_frozen_lpcr_passes_bce_gradient_to_latent():
    model = SparseResidualStage2(diffusion_target="epsilon").eval()
    model.lpcr.requires_grad_(False)
    hierarchy = batched_hierarchy(
        [torch.tensor([[10, 20, 30], [10, 20, 31]], dtype=torch.int32)], "cpu")
    latent = torch.zeros((len(hierarchy[-1]), LATENT_CHANNELS),
                         requires_grad=True)
    logits, _ = model.lpcr(
        latent, hierarchy, torch.ones((len(hierarchy[0]), 1)))
    loss, _ = generated_occupancy_bce(
        logits, torch.tensor([True, False]),
        torch.ones(2, dtype=torch.bool))
    loss.backward()
    assert latent.grad is not None
    assert torch.isfinite(latent.grad).all()
    assert all(parameter.grad is None for parameter in model.lpcr.parameters())


def test_fixed_validation_bce_matches_final_logits_and_counts_additions(monkeypatch):
    fine = torch.tensor([[0, 0, 80, 50], [0, 0, 80, 51]])
    logits = torch.tensor([0.5, 1.0])
    monkeypatch.setattr(evaluator, "predict_batch", lambda *args, **kwargs: {
        "fine": fine, "logits": logits})
    monkeypatch.setattr(evaluator, "reference_clean", lambda *args: (
        torch.tensor([[10.1, 0.1, -1.875]])))
    rows = [[{"frame_id": "00001", "faulty_xyz": torch.zeros((0, 3))}]]
    report = evaluator.fixed_threshold_validation(
        None, rows, None, torch.ones(2), device="cpu", seed=1, steps=2,
        generated_bce=True)
    expected = F.binary_cross_entropy_with_logits(
        logits, torch.tensor([1.0, 0.0]))
    assert report["generated_bce"]["loss"] == pytest.approx(float(expected))
    assert report["generated_bce"]["positive_count"] == 1
    assert report["generated_bce"]["negative_count"] == 1
    assert report["added"] == report["candidate_cells"] == 2
    assert "generated_decoder_loss" not in report


def test_bce_epoch_log_keeps_mse_and_occupancy_metrics():
    full = {"generated_bce": {"loss": 0.4}, "added": 25,
            "fault_area": {"faulty": {"iou": 0.2, "iou_0p2m": 0.3},
                           "completed": {"iou": 0.4, "iou_0p2m": 0.5}},
            "faulty_baseline_iou": 0.6, "faulty_baseline_iou_0p2m": 0.6,
            "iou": 0.62, "iou_0p2m": 0.63,
            "full_iou_improvement_pp": 2.0,
            "full_iou_0p2m_improvement_pp": 3.0}
    log = mse_epoch_log(5, 0.2, 0.3, full, 0.62,
                        train_bce=0.5, train_total=0.25, bce_weight=0.1)
    for value in ("train_noise_MSE=0.2000", "train_BCE=0.5000",
                  "train_total_loss=0.2500", "val_noise_MSE=0.3000",
                  "val_BCE=0.4000", "val_total_loss=0.3400",
                  "fault_area_exact_IoU=20.00%->40.00%",
                  "full_exact_IoU=60.00%->62.00% (+2.00 pp)",
                  "additions@0.5=25"):
        assert value in log
    skipped = mse_epoch_log(2, 0.2, 0.3, None, 0.62,
                            train_bce=0.5, train_total=0.25, bce_weight=0.1)
    assert "val_BCE=skipped" in skipped

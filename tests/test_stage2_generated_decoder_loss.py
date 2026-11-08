"""Gaussian-start occupancy supervision updates the denoiser, not LPCR."""

import pytest
import torch
from torch import nn

from models.radar_lidar_stage2.clean_denoising import batched_hierarchy
from models.radar_lidar_stage2.conditional_sampling import (
    sample_from_noise, sample_from_noise_truncated)
from models.radar_lidar_stage2.decoder_occupancy_loss import (
    asymmetric_occupancy_loss)
from models.radar_lidar_stage2.sparse_residual import (
    LATENT_CHANNELS, SparseResidualStage2)
from scripts import train_stage2_clean_denoiser as trainer
from scripts import evaluate_stage2_conditional_reconstruction as evaluator


def test_training_rollout_matches_inference_and_tracks_final_steps():
    class Diffusion(nn.Module):
        def __init__(self):
            super().__init__()
            self.scale = nn.Parameter(torch.tensor(1.0))
            self.gradient_flags = []

        def forward(self, noisy, faulty, radar, probability, present,
                    timestep, coords):
            self.gradient_flags.append(torch.is_grad_enabled())
            return self.scale * noisy + 0.05 * radar

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.diffusion = Diffusion()

    model = Model()
    coarse = torch.tensor([[0, 0, 0, 0], [0, 0, 0, 1]])
    faulty = torch.zeros((2, LATENT_CHANNELS))
    radar = torch.ones_like(faulty)
    conditions = (faulty, torch.ones(2, dtype=torch.bool), radar)
    alpha_bars = torch.linspace(0.999, 0.001, 1000)
    expected = sample_from_noise(
        model, coarse, conditions, alpha_bars, ["frame"], steps=4, seed=17)
    assert model.diffusion.gradient_flags == [False] * 4
    model.diffusion.gradient_flags.clear()
    generated = sample_from_noise_truncated(
        model, coarse, conditions, alpha_bars, ["frame"],
        steps=4, seed=17, gradient_steps=2)
    assert torch.equal(generated.detach(), expected)
    assert model.diffusion.gradient_flags == [False, False, True, True]
    generated.sum().backward()
    assert model.diffusion.scale.grad is not None
    assert torch.isfinite(model.diffusion.scale.grad)


def test_false_positive_weight_acts_on_decoder_logits():
    logits = torch.tensor([-0.5, 1.0, -1.0], requires_grad=True)
    occupied = torch.tensor([True, False, False])
    allowed = torch.tensor([True, True, False])
    mild, parts = asymmetric_occupancy_loss(
        logits, occupied, allowed, false_positive_weight=1.0)
    conservative, _ = asymmetric_occupancy_loss(
        logits, occupied, allowed, false_positive_weight=2.0)
    assert parts["positive_count"] == parts["negative_count"] == 1
    assert conservative > mild
    conservative.backward()
    assert logits.grad[1] > 0  # Penalize the false-positive logit.
    assert logits.grad[0] < 0  # Still reward the clean occupied cell.
    assert logits.grad[2] == 0  # Observed cells are not addition candidates.


def test_empty_positive_batch_still_penalizes_false_positives():
    logits = torch.tensor([1.0, -1.0], requires_grad=True)
    loss, parts = asymmetric_occupancy_loss(
        logits, torch.tensor([False, False]),
        torch.tensor([True, False]), false_positive_weight=2.0)
    assert parts["positive_count"] == 0
    assert parts["negative_count"] == 1
    assert torch.isfinite(loss)
    loss.backward()
    assert logits.grad[0] > 0
    assert logits.grad[1] == 0


def test_generated_loss_reaches_denoiser_through_frozen_decoder(monkeypatch):
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
    result = trainer._run_epoch(
        model, [[row]], optimizer, torch.linspace(0.999, 0.001, 1000),
        device="cpu", epoch=1, eval_timestep=499, seed=3103,
        condition_faulty_radar=True, train_radar_adapter=True,
        decoder_occupancy=trainer.DecoderOccupancyConfig(
            rollout_steps=4, gradient_steps=2, loss_weight=0.1,
            false_positive_weight=2.0))
    assert result["decoder_positive_sites"] == 1
    assert result["decoder_negative_sites"] == 1
    assert result["total_loss"] == pytest.approx(
        result["noise_mse"] + 0.1 * result["decoder_occupancy_loss"])
    assert torch.isfinite(model.diffusion.scale.grad)
    assert model.radar_adapter.weight.grad is not None
    assert model.lpcr_scale.grad is None


def test_real_frozen_lpcr_passes_margin_gradient_to_latent():
    model = SparseResidualStage2(diffusion_target="epsilon").eval()
    model.lpcr.requires_grad_(False)
    fine = torch.tensor([[10, 20, 30], [10, 20, 31]], dtype=torch.int32)
    hierarchy = batched_hierarchy([fine], "cpu")
    latent = torch.zeros((len(hierarchy[-1]), LATENT_CHANNELS),
                         requires_grad=True)
    logits, _ = model.lpcr(
        latent, hierarchy, torch.ones((len(hierarchy[0]), 1)))
    loss, _ = asymmetric_occupancy_loss(
        logits, torch.tensor([True, False]),
        torch.ones(2, dtype=torch.bool), false_positive_weight=2.0)
    loss.backward()
    assert latent.grad is not None
    assert torch.isfinite(latent.grad).all()
    assert all(parameter.grad is None for parameter in model.lpcr.parameters())


def test_real_sparse_denoiser_receives_generated_decoder_gradient():
    model = SparseResidualStage2(diffusion_target="epsilon").eval()
    model.requires_grad_(False)
    model.diffusion.requires_grad_(True)
    fine = torch.tensor([[10, 20, 30], [10, 20, 31]], dtype=torch.int32)
    hierarchy = batched_hierarchy([fine], "cpu")
    coarse = hierarchy[-1]
    faulty = torch.zeros((len(coarse), LATENT_CHANNELS))
    conditions = (faulty, torch.zeros(len(coarse), dtype=torch.bool),
                  torch.zeros_like(faulty))
    generated = sample_from_noise_truncated(
        model, coarse, conditions,
        torch.linspace(0.999, 0.9, 1000), ["frame"],
        steps=4, seed=17, gradient_steps=2)
    logits, _ = model.lpcr(
        generated, hierarchy, torch.ones((len(hierarchy[0]), 1)))
    loss, _ = asymmetric_occupancy_loss(
        logits, torch.tensor([True, False]),
        torch.ones(2, dtype=torch.bool), false_positive_weight=2.0)
    loss.backward()
    gradients = [parameter.grad for parameter in model.diffusion.parameters()
                 if parameter.grad is not None]
    assert gradients
    assert all(torch.isfinite(gradient).all() for gradient in gradients)
    assert all(parameter.grad is None for parameter in model.lpcr.parameters())


def test_validation_sweep_measures_generated_decoder_loss(monkeypatch):
    fine = torch.tensor([[0, 0, 80, 50], [0, 0, 80, 51]])
    logits = torch.tensor([0.5, 1.0])
    monkeypatch.setattr(evaluator, "predict_batch", lambda *args, **kwargs: {
        "fine": fine, "logits": logits})
    monkeypatch.setattr(evaluator, "reference_clean", lambda *args: (
        torch.tensor([[10.1, 0.1, -1.875]])))
    rows = [[{"frame_id": "00001", "faulty_xyz": torch.zeros((0, 3))}]]
    report, _ = evaluator.sweep(
        None, rows, ["00001"], None, torch.ones(2),
        device="cpu", seed=1, steps=2, decoder_loss_fp_weight=2.0)
    parts = report["generated_decoder_loss"]["calibration"]
    expected, _ = asymmetric_occupancy_loss(
        logits, torch.tensor([True, False]),
        torch.ones(2, dtype=torch.bool), false_positive_weight=2.0)
    assert parts["positive_count"] == parts["negative_count"] == 1
    assert parts["loss"] == pytest.approx(float(expected))


def test_periodic_validation_uses_only_fixed_threshold(monkeypatch):
    fine = torch.tensor([[0, 0, 80, 50], [0, 0, 80, 51]])
    logits = torch.tensor([1.0, 1.0])
    monkeypatch.setattr(evaluator, "predict_batch", lambda *args, **kwargs: {
        "fine": fine, "logits": logits})
    monkeypatch.setattr(evaluator, "reference_clean", lambda *args: (
        torch.tensor([[10.1, 0.1, -1.875]])))
    rows = [[{"frame_id": "00001", "faulty_xyz": torch.zeros((0, 3))}]]
    report = evaluator.fixed_threshold_validation(
        None, rows, None, torch.ones(2), device="cpu", seed=1, steps=2,
        decoder_loss_fp_weight=2.0)
    assert report["threshold"] == 0.5
    assert (report["tp"], report["predicted"], report["target"]) == (1, 2, 1)
    assert report["iou"] == pytest.approx(0.5)
    assert report["iou_0p2m"] == pytest.approx(0.5)
    assert report["faulty_baseline_iou"] == 0.0
    assert report["generated_decoder_loss"]["loss"] == pytest.approx(3.0)

"""The clean-start trial uses only epsilon MSE and ordinary occupancy BCE."""

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from models.radar_lidar_stage2.clean_denoising import batched_hierarchy
from models.radar_lidar_stage2.sparse_residual import (
    LATENT_CHANNELS, SparseResidualStage2)
from scripts import train_stage2_clean_denoiser as trainer


def test_ordinary_bce_reaches_denoiser_through_frozen_lpcr(monkeypatch):
    class Diffusion(nn.Module):
        def __init__(self):
            super().__init__()
            self.scale = nn.Parameter(torch.tensor(0.25))
            self.timesteps = []

        def forward(self, noisy, faulty, radar, probability, present,
                    timestep, coords):
            self.timesteps.append(int(timestep[0]))
            return self.scale * noisy

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.diffusion = Diffusion()
            self.lpcr_scale = nn.Parameter(torch.tensor(0.7),
                                           requires_grad=False)
            self.last_logits = None

        def lidar_encoder(self, clean, coarse, point_ids):
            return torch.full((len(coarse), 2), 0.25), torch.ones(
                len(coarse), dtype=torch.bool)

        def lpcr(self, latent, hierarchy, site_features):
            self.last_logits = self.lpcr_scale * latent[0, 0] + torch.tensor(
                [0.0, -0.5])
            return self.last_logits, None

    fine = torch.tensor([[0, 0, 0, 0], [0, 0, 0, 1]])
    coarse = fine[:1]
    monkeypatch.setattr(trainer, "_batch", lambda rows, device: (
        (fine, fine, fine, fine, coarse), torch.zeros((1, 3)),
        torch.zeros(1, dtype=torch.long)))
    target = torch.tensor([True, False])
    monkeypatch.setattr(trainer, "offset_targets", lambda *args: (
        target, None, None))
    model = Model()
    optimizer = torch.optim.SGD(model.diffusion.parameters(), lr=0)
    result = trainer._run_epoch(
        model, [[{"frame_id": "dummy"}]], optimizer,
        torch.linspace(0.999, 0.001, 1000), device="cpu", epoch=1,
        eval_timestep=499, seed=3103, occupancy_bce=True)

    assert len(model.diffusion.timesteps) == 2
    assert model.diffusion.timesteps[1] == 499
    assert result["occupancy_bce"] == pytest.approx(
        F.binary_cross_entropy_with_logits(
            model.last_logits.detach(), target.float()).item())
    assert result["total_loss"] == pytest.approx(
        result["noise_mse"] + result["occupancy_bce"])
    assert model.diffusion.scale.grad is not None
    assert torch.isfinite(model.diffusion.scale.grad)
    assert model.lpcr_scale.grad is None


def test_actual_frozen_lpcr_passes_occupancy_gradient_to_latent():
    model = SparseResidualStage2(diffusion_target="epsilon").eval()
    model.lpcr.requires_grad_(False)
    fine = torch.tensor([[10, 20, 30], [10, 20, 31]], dtype=torch.int32)
    hierarchy = batched_hierarchy([fine], "cpu")
    latent = torch.zeros((len(hierarchy[-1]), LATENT_CHANNELS),
                         requires_grad=True)
    logits, _ = model.lpcr(
        latent, hierarchy, torch.ones((len(hierarchy[0]), 1)))
    F.binary_cross_entropy_with_logits(
        logits, torch.tensor([1.0, 0.0])).backward()
    assert latent.grad is not None
    assert torch.isfinite(latent.grad).all()
    assert all(parameter.grad is None for parameter in model.lpcr.parameters())

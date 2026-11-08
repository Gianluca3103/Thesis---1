"""Validation noise loss is repeatable and avoids the occupancy decoder."""

import torch
from torch import nn

from scripts import train_stage2_clean_denoiser as trainer


def test_validation_noise_mse_uses_training_timestep_distribution(monkeypatch):
    class Diffusion(nn.Module):
        def __init__(self):
            super().__init__()
            self.timesteps = []

        def forward(self, noisy, faulty, radar, probability, present,
                    timestep, coords):
            assert not torch.is_grad_enabled()
            self.timesteps.append(int(timestep[0]))
            return noisy * 0.2 + radar * 0.1

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.diffusion = Diffusion()

        def lidar_encoder(self, clean, coarse, point_ids):
            return torch.full((len(coarse), 2), 0.25), torch.ones(
                len(coarse), dtype=torch.bool)

        def lpcr(self, *args):
            raise AssertionError("Loss-only validation must not decode occupancy")

    fine = torch.tensor([[0, 0, 0, 0]])
    hierarchy = (fine, fine, fine, fine, fine)
    monkeypatch.setattr(trainer, "_batch", lambda rows, device: (
        hierarchy, torch.zeros((1, 3)), torch.zeros(1, dtype=torch.long)))
    monkeypatch.setattr(trainer, "_conditioned_inputs", lambda *args, **kwargs: (
        torch.ones((1, 2)), torch.ones(1, dtype=torch.bool),
        torch.ones((1, 2))))
    model = Model()
    loader = [[{"frame_id": "a"}], [{"frame_id": "b"}]]
    alpha_bars = torch.linspace(0.999, 0.001, 1000)

    def evaluate(epoch):
        model.diffusion.timesteps.clear()
        result = trainer._run_epoch(
            model, loader, None, alpha_bars, device="cpu", epoch=epoch,
            eval_timestep=499, seed=3103, condition_faulty_radar=True,
            eval_random_timesteps=True, eval_loss_only=True)
        return result, model.diffusion.timesteps.copy()

    first, steps = evaluate(1)
    second, repeated_steps = evaluate(37)
    assert first["frames"] == 2
    assert torch.isfinite(torch.tensor(first["noise_mse"]))
    assert first["noise_mse"] == second["noise_mse"]
    assert steps == repeated_steps
    assert steps != [499, 499]
    assert "eligible_occupancy" not in first

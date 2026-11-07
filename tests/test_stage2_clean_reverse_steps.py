"""Check the full-noise DDIM schedule and the clean-latent target algebra."""

import torch
from torch import nn

from scripts import train_stage2_clean_denoiser as trainer
from scripts.train_stage2_clean_denoiser import reverse_schedule, reverse_step


def test_train_and_validation_schedules_share_start_middle_and_endpoint():
    train = reverse_schedule(5)
    val = reverse_schedule(3)
    assert train.tolist() == [999, 799, 599, 399, 199]
    assert val.tolist() == [999, 599, 199]
    assert torch.equal(train[::2], val)


def test_reverse_step_recovers_clean_latent_with_exact_epsilon():
    x0 = torch.tensor([[-0.8, 0.2], [0.5, 0.9]])
    epsilon = torch.tensor([[0.7, -0.3], [-0.2, 0.4]])
    alpha = torch.tensor(0.4)
    next_alpha = torch.tensor(0.7)
    xt = alpha.sqrt() * x0 + (1 - alpha).sqrt() * epsilon
    recovered, following = reverse_step(xt, epsilon, alpha, next_alpha)
    assert torch.allclose(recovered, x0, atol=1e-6)
    assert torch.allclose(
        following, next_alpha.sqrt() * x0 + (1 - next_alpha).sqrt() * epsilon,
        atol=1e-6)
    terminal, following = reverse_step(xt, epsilon, alpha, None)
    assert following is None
    assert torch.allclose(terminal, x0, atol=1e-6)


def test_rolled_out_epsilon_target_matches_actual_state():
    # The target for an on-policy DDIM state must be derived from that state.
    alpha = torch.tensor(0.75)
    x0 = torch.tensor([[0.3, -0.4]])
    rolled_state = torch.tensor([[0.7, 0.1]])
    target = (rolled_state - alpha.sqrt() * x0) / (1 - alpha).sqrt()
    recovered, _ = reverse_step(rolled_state, target, alpha, None)
    assert torch.allclose(recovered, x0, atol=1e-6)


def test_training_uses_five_reverse_passes_and_validation_uses_three(monkeypatch):
    class Diffusion(nn.Module):
        def __init__(self):
            super().__init__()
            self.scale = nn.Parameter(torch.tensor(0.1))
            self.timesteps = []

        def forward(self, noisy, faulty, radar, probability, present,
                    timestep, coords):
            assert torch.count_nonzero(faulty) == 0
            assert torch.count_nonzero(radar) == 0
            self.timesteps.append(int(timestep[0]))
            return self.scale * noisy

    class Model:
        def __init__(self):
            self.diffusion = Diffusion()

        def eval(self):
            return self

        def lidar_encoder(self, clean, coarse, point_ids):
            return torch.full((len(coarse), 2), 0.25), torch.ones(
                len(coarse), dtype=torch.bool)

        def lpcr(self, latent, hierarchy, site_features):
            return latent[:, 0], None

    fine = torch.tensor([[0, 0, 0, 0]])
    hierarchy = (fine, fine)
    monkeypatch.setattr(trainer, "_batch", lambda rows, device: (
        hierarchy, torch.zeros((1, 3)), torch.zeros(1, dtype=torch.long)))
    monkeypatch.setattr(trainer, "offset_targets", lambda *args: (
        torch.ones(1, dtype=torch.bool), None, None))
    model = Model()
    optimizer = torch.optim.SGD(model.diffusion.parameters(), lr=0.01)
    alpha_bars = torch.linspace(0.999, 0.001, 1000)
    rows = [[{"frame_id": "dummy"}]]

    train = trainer._run_epoch(
        model, rows, optimizer, alpha_bars, device="cpu", epoch=1,
        eval_timestep=499, seed=3103, reverse_train_steps=5,
        reverse_val_steps=3)
    assert model.diffusion.timesteps == [999, 799, 599, 399, 199]
    assert train["reverse_steps"] == 5
    assert torch.isfinite(model.diffusion.scale)

    model.diffusion.timesteps.clear()
    val = trainer._run_epoch(
        model, rows, None, alpha_bars, device="cpu", epoch=1,
        eval_timestep=499, seed=3103, reverse_train_steps=5,
        reverse_val_steps=3)
    assert model.diffusion.timesteps == [999, 599, 199]
    assert val["reverse_steps"] == 3
    assert val["eligible_occupancy"]["target"] == 1

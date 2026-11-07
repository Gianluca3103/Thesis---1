"""Check the full-noise DDIM schedule and the clean-latent target algebra."""

import torch
from torch import nn

from models.radar_lidar_crossmodal_v1.radar_pruner import (
    radar_support_mask, supported_coordinates)
from models.radar_lidar_stage2.sparse_geometry import GRID
from models.radar_lidar_stage2.sparse_residual import (
    LATENT_CHANNELS, SparseResidualStage2)
from scripts import train_stage2_clean_denoiser as trainer
from scripts.train_stage2_clean_denoiser import (
    CONDITIONED_ONE_STEP_REVISION, REVISION, REVERSE_REVISION,
    resolve_sampling_strategy,
    reverse_schedule, reverse_step, sample_clean_latent)


def test_train_and_validation_schedules_share_start_middle_and_endpoint():
    train = reverse_schedule(5)
    val = reverse_schedule(3)
    assert train.tolist() == [999, 799, 599, 399, 199]
    assert val.tolist() == [999, 599, 199]
    assert torch.equal(train[::2], val)


def test_crossed_sampler_uses_the_requested_timesteps():
    class Diffusion(nn.Module):
        def __init__(self):
            super().__init__()
            self.timesteps = []

        def forward(self, noisy, faulty, radar, probability, present,
                    timestep, coords):
            self.timesteps.append(int(timestep[0]))
            assert not faulty.any() and not radar.any()
            return torch.zeros_like(noisy)

    class Model:
        diffusion = Diffusion()

    assert resolve_sampling_strategy(REVISION, "checkpoint") == "one_step"
    assert resolve_sampling_strategy(REVERSE_REVISION, "checkpoint") == "reverse3"
    assert resolve_sampling_strategy(REVISION, "reverse3") == "reverse3"
    assert resolve_sampling_strategy(REVERSE_REVISION, "one_step") == "one_step"
    assert resolve_sampling_strategy(
        CONDITIONED_ONE_STEP_REVISION, "checkpoint") == "one_step"
    model = Model()
    x0 = torch.full((1, 2), 0.25)
    coarse = torch.tensor([[0, 0, 0, 0]])
    alpha_bars = torch.linspace(0.999, 0.001, 1000)
    generator = torch.Generator().manual_seed(3103)
    one = sample_clean_latent(
        model, x0, coarse, alpha_bars, generator, batch_size=1,
        strategy="one_step", eval_timestep=499)
    assert one.shape == x0.shape
    assert model.diffusion.timesteps == [499]
    model.diffusion.timesteps.clear()
    reverse = sample_clean_latent(
        model, x0, coarse, alpha_bars, generator, batch_size=1,
        strategy="reverse3", eval_timestep=499)
    assert reverse.shape == x0.shape
    assert model.diffusion.timesteps == [999, 599, 199]
    same_noise = torch.Generator().manual_seed(123)
    first = sample_clean_latent(
        model, x0, coarse, alpha_bars, same_noise, batch_size=1,
        strategy="reverse3", eval_timestep=499)
    same_noise = torch.Generator().manual_seed(123)
    different_clean = sample_clean_latent(
        model, x0 + 0.5, coarse, alpha_bars, same_noise, batch_size=1,
        strategy="reverse3", eval_timestep=499)
    assert torch.equal(first, different_clean)


def test_one_step_sampling_uses_both_condition_tensors():
    class Diffusion(nn.Module):
        def forward(self, noisy, faulty, radar, probability, present,
                    timestep, coords):
            return faulty * present[:, None] + radar

    class Model:
        diffusion = Diffusion()

    model = Model()
    x0 = torch.zeros((1, 2))
    coarse = torch.tensor([[0, 0, 0, 0]])
    alpha_bars = torch.linspace(0.999, 0.001, 1000)
    def sample(conditions):
        return sample_clean_latent(
            model, x0, coarse, alpha_bars,
            torch.Generator().manual_seed(17), batch_size=1,
            strategy="one_step", eval_timestep=499,
            conditions=conditions)
    gate_only = sample(None)
    faulty = torch.full((1, 2), 0.4)
    radar = torch.full((1, 2), 0.6)
    conditioned = sample((faulty, torch.ones(1, dtype=torch.bool), radar))
    assert not torch.equal(gate_only, conditioned)
    assert not torch.equal(
        conditioned, sample((torch.zeros_like(faulty),
                             torch.ones(1, dtype=torch.bool), radar)))
    assert not torch.equal(
        conditioned, sample((faulty, torch.ones(1, dtype=torch.bool),
                             torch.zeros_like(radar))))


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


def test_faulty_and_radar_conditions_align_to_full_support_gate():
    class Model:
        grid = GRID
        conditions = SparseResidualStage2.conditions

        def __init__(self):
            self.radar_adapter = nn.Linear(64, LATENT_CHANNELS, bias=False)
            with torch.no_grad():
                self.radar_adapter.weight.zero_()
                self.radar_adapter.weight[:, 0] = 1

        def lidar_encoder(self, faulty, coarse, point_ids):
            assert len(faulty) == 1
            return torch.ones((len(coarse), LATENT_CHANNELS)), torch.ones(
                len(coarse), dtype=torch.bool)

    radar = torch.zeros((12, 7))
    radar[:, 0] = 10.05 + 0.2 * torch.arange(12)
    radar[:, 1:3] = 0.05
    mask = radar_support_mask(radar, 1.6, 12)
    fine = supported_coordinates(mask)
    hierarchy = trainer.batched_hierarchy([fine], "cpu")
    row = {"radar": radar, "faulty_xyz": torch.tensor([[10.1, 0.1, 0.1]]),
           "support_mask": mask, "fine_zyx": fine}
    model = Model()
    faulty, present, measured_radar = trainer._conditioned_inputs(
        model, [row], hierarchy, "cpu", train_radar_adapter=True)
    assert faulty.shape == measured_radar.shape
    assert present.all()
    assert bool((measured_radar > 0).any())
    measured_radar.mean().backward()
    assert model.radar_adapter.weight.grad is not None
    assert bool((model.radar_adapter.weight.grad.abs() > 0).any())


def test_training_uses_five_reverse_passes_and_validation_uses_three(monkeypatch):
    class Diffusion(nn.Module):
        def __init__(self):
            super().__init__()
            self.scale = nn.Parameter(torch.tensor(0.1))
            self.timesteps = []
            self.expect_conditioned = False

        def forward(self, noisy, faulty, radar, probability, present,
                    timestep, coords):
            if self.expect_conditioned:
                assert torch.count_nonzero(faulty) > 0
                assert torch.count_nonzero(radar) > 0
                assert present.all()
            else:
                assert torch.count_nonzero(faulty) == 0
                assert torch.count_nonzero(radar) == 0
            self.timesteps.append(int(timestep[0]))
            return self.scale * noisy + 0.1 * radar

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.diffusion = Diffusion()
            self.radar_adapter = nn.Linear(2, 2, bias=False)

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

    monkeypatch.setattr(trainer, "_conditioned_inputs", lambda model, *args, **kwargs: (
        torch.full((1, 2), 0.4), torch.ones(1, dtype=torch.bool),
        model.radar_adapter(torch.full((1, 2), 0.6))))
    model.diffusion.expect_conditioned = True
    model.diffusion.timesteps.clear()
    conditioned = trainer._run_epoch(
        model, rows, None, alpha_bars, device="cpu", epoch=1,
        eval_timestep=499, seed=3103, reverse_train_steps=5,
        reverse_val_steps=3, condition_faulty_radar=True)
    assert model.diffusion.timesteps == [999, 599, 199]
    assert conditioned["conditioning"] == "faulty_LiDAR_and_measured_radar"

    before = model.radar_adapter.weight.detach().clone()
    model.diffusion.timesteps.clear()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    single = trainer._run_epoch(
        model, rows, optimizer, alpha_bars, device="cpu", epoch=1,
        eval_timestep=499, seed=3103, condition_faulty_radar=True,
        train_radar_adapter=True)
    assert len(model.diffusion.timesteps) == 1
    assert single["conditioning"] == "faulty_LiDAR_and_measured_radar"
    assert not torch.equal(model.radar_adapter.weight.detach(), before)

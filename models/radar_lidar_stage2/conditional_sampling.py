"""Conditional Stage-II sampling from noise without a clean-LiDAR input."""

from __future__ import annotations

import hashlib

import torch

from .sparse_residual import LATENT_CHANNELS


def sampling_schedule(steps: int, total: int = 1000) -> torch.Tensor:
    """Descending DDIM timesteps including the last clean-side timestep."""
    if steps < 2 or steps > total:
        raise ValueError("Sampling steps must be in [2, total timesteps]")
    schedule = torch.linspace(total - 1, 0, steps).round().long()
    if not bool((schedule[:-1] > schedule[1:]).all()):
        raise ValueError("Sampling timesteps must be distinct")
    return schedule


def _frame_seed(seed: int, frame_id: str) -> int:
    digest = hashlib.blake2b(
        f"{seed}:{frame_id}".encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "little") % (2**63)


def _seeded_noise(template: torch.Tensor, coarse: torch.Tensor,
                  frame_ids: list[str], seed: int) -> torch.Tensor:
    noise = torch.empty_like(template)
    for index, frame_id in enumerate(frame_ids):
        selected = coarse[:, 0] == index
        generator = torch.Generator(device=template.device).manual_seed(
            _frame_seed(seed, frame_id))
        noise[selected] = torch.randn(
            (int(selected.sum()), LATENT_CHANNELS),
            device=template.device, dtype=template.dtype, generator=generator)
    return noise


def faulty_to_clean_noise_target(
        faulty: torch.Tensor, clean: torch.Tensor, epsilon: torch.Tensor,
        alpha: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Noisy faulty input and noise-equivalent target for clean restoration.

    The target includes the faulty-to-clean correction; it is not pure epsilon.
    """
    if faulty.shape != clean.shape or faulty.shape != epsilon.shape:
        raise ValueError("Faulty, clean, and noise latents must align")
    xt = alpha.sqrt() * faulty + (1 - alpha).sqrt() * epsilon
    target = (xt - alpha.sqrt() * clean) / (1 - alpha).sqrt()
    return xt, target


@torch.no_grad()
def sample_from_noise(model, coarse: torch.Tensor,
                      conditions: tuple[torch.Tensor, torch.Tensor,
                                        torch.Tensor],
                      alpha_bars: torch.Tensor,
                      frame_ids: list[str], *, steps: int,
                      seed: int) -> torch.Tensor:
    """Generate clean-latent estimates using faulty LiDAR and radar only.

    The starting Gaussian is deterministic per frame, independent of batch
    packing. Clean LiDAR is neither an argument nor read by this function.
    """
    faulty, present, radar = conditions
    count = len(coarse)
    if (coarse.ndim != 2 or coarse.shape[1] != 4 or
            faulty.shape != (count, LATENT_CHANNELS) or
            radar.shape != faulty.shape or present.shape != (count,) or
            alpha_bars.ndim != 1 or len(frame_ids) < 1):
        raise ValueError("Conditional latent sites and features differ")
    schedule = sampling_schedule(steps, len(alpha_bars)).to(coarse.device)
    if not count:
        return faulty.new_empty((0, LATENT_CHANNELS))
    if int(coarse[:, 0].max()) >= len(frame_ids):
        raise ValueError("Missing frame ID for a latent site")
    xt = _seeded_noise(faulty, coarse, frame_ids, seed)
    probability = faulty.new_ones((count, 1))
    for index, timestep in enumerate(schedule):
        predicted_epsilon = model.diffusion(
            xt, faulty, radar, probability, present,
            timestep.expand(len(frame_ids)), coarse)
        alpha = alpha_bars[timestep]
        x0_hat = ((xt - (1 - alpha).sqrt() * predicted_epsilon)
                  / alpha.sqrt()).clamp(-1, 1)
        if index + 1 == len(schedule):
            return x0_hat
        next_alpha = alpha_bars[schedule[index + 1]]
        xt = (next_alpha.sqrt() * x0_hat
              + (1 - next_alpha).sqrt() * predicted_epsilon)
    raise AssertionError("Unreachable empty sampling schedule")


@torch.no_grad()
def sample_from_faulty(model, coarse: torch.Tensor,
                       conditions: tuple[torch.Tensor, torch.Tensor,
                                         torch.Tensor],
                       alpha_bars: torch.Tensor, frame_ids: list[str], *,
                       timestep: int, seed: int) -> torch.Tensor:
    """One-step restoration from a noisy faulty-LiDAR latent, never clean."""
    faulty, present, radar = conditions
    count = len(coarse)
    if (coarse.ndim != 2 or coarse.shape[1] != 4 or
            faulty.shape != (count, LATENT_CHANNELS) or
            radar.shape != faulty.shape or present.shape != (count,) or
            alpha_bars.ndim != 1 or not 0 <= timestep < len(alpha_bars) or
            not frame_ids):
        raise ValueError("Faulty-start latent sites or timestep differ")
    if not count:
        return faulty.new_empty((0, LATENT_CHANNELS))
    if int(coarse[:, 0].max()) >= len(frame_ids):
        raise ValueError("Missing frame ID for a latent site")
    alpha = alpha_bars[timestep]
    noise = _seeded_noise(faulty, coarse, frame_ids, seed)
    xt, _ = faulty_to_clean_noise_target(faulty, faulty, noise, alpha)
    predicted = model.diffusion(
        xt, faulty, radar, faulty.new_ones((count, 1)), present,
        torch.full((len(frame_ids),), timestep, device=faulty.device,
                   dtype=torch.long), coarse)
    return ((xt - (1 - alpha).sqrt() * predicted) / alpha.sqrt()).clamp(-1, 1)

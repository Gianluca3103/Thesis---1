"""Sparse voxel autoencoder adapted from MinkowskiEngine's generative decoder demo.

The upstream demo starts from a CAD class label. Here a learned sparse encoder
starts from VoD LiDAR occupancy. Only coarse latent coordinates and features
enter the decoder; it receives no full-resolution coordinates or skip features.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import nn
from torch.nn import functional as F

try:
    import MinkowskiEngine as ME
except ImportError as exc:
    ME = None
    _ME_ERROR = exc

from models.radar_lidar_stage2.sparse_geometry import GRID


def _keys(coords: torch.Tensor) -> torch.Tensor:
    """Injective keys for valid BXYZ Stage-II coordinates."""
    x_size, y_size, z_size = GRID.shape_zyx[::-1]
    values = coords.long()
    return ((values[:, 0] * z_size + values[:, 3]) * y_size + values[:, 2]) * x_size + values[:, 1]


def _inside(coords: torch.Tensor) -> torch.Tensor:
    shape = coords.new_tensor(GRID.shape_zyx[::-1])
    return ((coords[:, 1:] >= 0) & (coords[:, 1:] < shape)).all(dim=1)


def _targets_at(candidates: torch.Tensor, fine: torch.Tensor, stride: int) -> torch.Tensor:
    coarse = fine.clone()
    coarse[:, 1:] = torch.div(coarse[:, 1:], stride, rounding_mode="floor") * stride
    target_keys = torch.unique(_keys(coarse))
    return _inside(candidates) & torch.isin(_keys(candidates), target_keys)


@dataclass
class Reconstruction:
    coordinates: torch.Tensor  # [N,4] BXYZ candidates, independent of target coordinates
    logits: torch.Tensor  # [N] occupancy scores on candidates
    loss: torch.Tensor | None
    latent_sites: int
    reachable_targets: int | None


class SparseVoxelAutoencoder(nn.Module):
    def __init__(self, *, width: int = 32, levels: int = 1):
        super().__init__()
        if ME is None:
            raise RuntimeError("MinkowskiEngine is required for this trial") from _ME_ERROR
        if width < 8 or levels not in (1, 2):
            raise ValueError("Use width >= 8 and one or two stride-2 levels")
        self.width, self.levels = width, levels
        encoder = [ME.MinkowskiConvolution(1, width, kernel_size=3, dimension=3),
                   ME.MinkowskiBatchNorm(width), ME.MinkowskiReLU()]
        for level in range(levels):
            cin, cout = width * 2**level, width * 2**(level + 1)
            encoder.extend((
                ME.MinkowskiConvolution(cin, cout, kernel_size=2, stride=2, dimension=3),
                ME.MinkowskiBatchNorm(cout), ME.MinkowskiReLU(),
                ME.MinkowskiConvolution(cout, cout, kernel_size=3, dimension=3),
                ME.MinkowskiBatchNorm(cout), ME.MinkowskiReLU(),
            ))
        self.encoder = nn.Sequential(*encoder)
        self.decoder = nn.ModuleList()
        self.classifier = nn.ModuleList()
        for level in reversed(range(levels)):
            cin, cout = width * 2**(level + 1), width * 2**level
            self.decoder.append(nn.Sequential(
                ME.MinkowskiGenerativeConvolutionTranspose(
                    cin, cout, kernel_size=2, stride=2, dimension=3),
                ME.MinkowskiBatchNorm(cout), ME.MinkowskiReLU(),
                ME.MinkowskiConvolution(cout, cout, kernel_size=3, dimension=3),
                ME.MinkowskiBatchNorm(cout), ME.MinkowskiReLU(),
            ))
            self.classifier.append(ME.MinkowskiConvolution(
                cout, 1, kernel_size=1, bias=True, dimension=3))
        self.pruning = ME.MinkowskiPruning()

    def encode(self, fine: torch.Tensor):
        if fine.ndim != 2 or fine.shape[1] != 4 or not bool(_inside(fine).all()):
            raise ValueError("Expected in-grid BXYZ LiDAR voxel coordinates")
        features = torch.ones((len(fine), 1), dtype=torch.float32, device=fine.device)
        source = ME.SparseTensor(features=features, coordinates=fine.int())
        latent = self.encoder(source)
        # A fresh coordinate manager prevents the decoder from consulting the
        # input-resolution coordinate map left behind by the encoder.
        return ME.SparseTensor(features=latent.F, coordinates=latent.C,
                               tensor_stride=latent.tensor_stride)

    def decode(self, latent, *, fine_targets: torch.Tensor | None = None,
               threshold: float = 0.5) -> Reconstruction:
        if not 0 < threshold < 1:
            raise ValueError("Occupancy threshold must be strictly between 0 and 1")
        state = latent
        losses = []
        reachable = None
        cutoff = math.log(threshold / (1 - threshold))
        for index, (block, classifier) in enumerate(zip(self.decoder, self.classifier)):
            state = block(state)
            logits = classifier(state).F[:, 0]
            coordinates = state.C
            if fine_targets is not None:
                labels = _targets_at(coordinates, fine_targets, state.tensor_stride[0])
                losses.append(F.binary_cross_entropy_with_logits(logits, labels.float()))
                if index == self.levels - 1:
                    reachable = int(labels.sum().item())
            if index < self.levels - 1:
                keep = _inside(coordinates) & (logits.detach() >= cutoff)
                if self.training and fine_targets is not None:
                    keep |= labels
                if not bool(keep.any()):
                    valid = _inside(coordinates)
                    if not bool(valid.any()):
                        raise RuntimeError("Generative decoder produced no in-grid candidates")
                    keep[torch.argmax(logits.detach().masked_fill(~valid, -torch.inf))] = True
                state = self.pruning(state, keep)
        return Reconstruction(coordinates, logits,
                              torch.stack(losses).mean() if losses else None,
                              latent.F.shape[0], reachable)

    def forward(self, fine: torch.Tensor, *, threshold: float = 0.5) -> Reconstruction:
        latent = self.encode(fine)
        # Targets supervise BCE only. In eval mode they never force a decoder
        # coordinate to survive; inference can call decode(latent) without them.
        return self.decode(latent, fine_targets=fine, threshold=threshold)

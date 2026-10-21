"""LiDAR encoder and LPCR decoder adapted from R2LDM's first training stage.

Source architecture: https://github.com/NathanDrake67/R2LDM at commit
4eac0a2c5a8a9c9aca44345a1d264fb184cb028f (Apache-2.0). The sparse
backbone follows ``SpMiddleResNetFHD``; the reconstruction path follows the
active ``S2D_RPN`` modules and the fine-scale Stage-0 loss. The unused RPN and
coarse reconstruction heads are intentionally absent. Fresh training is
required: checkpoints from R2LDM are not state-dict compatible with this port.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from .data import R2LDMGrid

try:
    import spconv.pytorch as spconv
except ImportError:
    spconv = None


_SparseModule = spconv.SparseModule if spconv is not None else nn.Module


class SparseResidualBlock(_SparseModule):
    def __init__(self, channels: int, indice_key: str):
        super().__init__()
        self.conv1 = spconv.SubMConv3d(channels, channels, 3, padding=1,
                                      indice_key=indice_key, bias=True)
        self.norm1 = nn.BatchNorm1d(channels, eps=1e-3, momentum=0.01)
        self.conv2 = spconv.SubMConv3d(channels, channels, 3, padding=1,
                                      indice_key=indice_key, bias=True)
        self.norm2 = nn.BatchNorm1d(channels, eps=1e-3, momentum=0.01)

    def forward(self, x):
        result = self.conv1(x)
        result = result.replace_feature(F.relu(self.norm1(result.features)))
        result = self.conv2(result)
        return result.replace_feature(F.relu(self.norm2(result.features) + x.features))


def _down(in_channels: int, out_channels: int, kernel, stride, padding, key: str):
    return spconv.SparseSequential(
        spconv.SparseConv3d(in_channels, out_channels, kernel, stride=stride,
                            padding=padding, bias=False),
        nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01),
        nn.ReLU(inplace=True),
        SparseResidualBlock(out_channels, key),
        SparseResidualBlock(out_channels, key),
    )


class R2LDMSparseLidarEncoder(nn.Module):
    """R2LDM mean-XYZ reader and 4-channel sparse residual backbone."""

    def __init__(self, grid: R2LDMGrid):
        super().__init__()
        if spconv is None:
            raise RuntimeError("R2LDM LiDAR LPCR requires spconv.pytorch and CUDA")
        grid.validate()
        self.grid = grid
        self.stem = spconv.SparseSequential(
            spconv.SubMConv3d(3, 4, 3, padding=1, bias=False, indice_key="res0"),
            nn.BatchNorm1d(4, eps=1e-3, momentum=0.01),
            nn.ReLU(inplace=True),
        )
        self.level1 = spconv.SparseSequential(
            SparseResidualBlock(4, "res0"), SparseResidualBlock(4, "res0"))
        self.level2 = _down(4, 4, 3, 2, 1, "res1")
        self.level3 = _down(4, 4, 3, 2, 1, "res2")
        self.level4 = _down(4, 4, 3, (2, 1, 1), (0, 1, 1), "res3")
        self.final = spconv.SparseSequential(
            spconv.SparseConv3d(4, 4, 1, stride=(2, 1, 1), bias=False),
            nn.BatchNorm1d(4, eps=1e-3, momentum=0.01),
            nn.ReLU(inplace=True),
        )

    def forward(self, means: torch.Tensor, coords: torch.Tensor,
                batch_size: int) -> torch.Tensor:
        if means.ndim != 2 or means.shape[1] != 3 or coords.shape != (len(means), 4):
            raise ValueError("Expected voxel mean XYZ [N,3] and BZYX [N,4]")
        if not len(means) or batch_size < 1:
            raise ValueError("Sparse LiDAR encoder needs occupied voxels")
        x = spconv.SparseConvTensor(means.contiguous(), coords.int().contiguous(),
                                   self.grid.shape_zyx, batch_size)
        x = self.level1(self.stem(x))
        x = self.level2(x)
        x = self.level3(x)
        x = self.level4(x)
        dense = self.final(x).dense()
        if dense.shape[1:] != (4, 2, 80, 80):
            raise RuntimeError(f"Unexpected R2LDM sparse encoder output {dense.shape}")
        return dense.reshape(batch_size, 8, 80, 80)


class _ConvNeXtBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(32, 32, 7, padding=3, groups=32),
            nn.LayerNorm((32, 20, 20), eps=1e-6),
            nn.Conv2d(32, 128, 1),
            nn.GELU(),
            nn.Conv2d(128, 32, 1),
        )

    def forward(self, x):
        return x + self.block(x)


class R2LDMPointReconstructionDecoder(nn.Module):
    """S2D latent map to fine occupancy logits and XYZ offsets."""

    def __init__(self):
        super().__init__()
        self.down1 = nn.Sequential(
            nn.Conv2d(8, 32, 2, stride=2), nn.BatchNorm2d(32), nn.GELU(),
            nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32), nn.GELU(),
        )
        self.down2 = nn.Sequential(
            nn.Conv2d(32, 32, 3, stride=2, padding=1), nn.BatchNorm2d(32), nn.GELU(),
            nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32), nn.GELU(),
        )
        self.convnext = nn.ModuleList(_ConvNeXtBlock() for _ in range(3))
        self.up2d = nn.Sequential(
            nn.ConvTranspose2d(32, 32, 4, stride=2, padding=1),
            nn.BatchNorm2d(32), nn.GELU(),
        )
        self.fuse2d = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.GELU(),
            nn.ConvTranspose2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128), nn.GELU(),
        )
        self.to_volume = nn.Sequential(
            nn.Conv2d(128, 640, 1), nn.BatchNorm2d(640), nn.GELU(),
        )
        self.up3d_1 = nn.Sequential(
            nn.Conv3d(128, 32, 1), nn.BatchNorm3d(32), nn.ReLU(),
            nn.ConvTranspose3d(32, 32, 4, stride=2, padding=1),
            nn.BatchNorm3d(32), nn.ReLU(),
        )
        self.up3d_2 = nn.Sequential(
            nn.Conv3d(32, 16, 1), nn.BatchNorm3d(16), nn.ReLU(),
            nn.ConvTranspose3d(16, 16, 4, stride=2, padding=1),
            nn.BatchNorm3d(16), nn.ReLU(),
        )
        self.up3d_3 = nn.Sequential(
            nn.Conv3d(16, 8, 1), nn.BatchNorm3d(8), nn.ReLU(),
            nn.ConvTranspose3d(8, 3, 4, stride=2, padding=1),
            nn.BatchNorm3d(3), nn.ReLU(),
        )
        self.mask_logits = nn.Conv3d(3, 2, 1)
        self.offset_raw = nn.Conv3d(3, 3, 1)

    def forward(self, latent: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if latent.ndim != 4 or latent.shape[1:] != (8, 80, 80):
            raise ValueError("R2LDM LPCR decoder expects [B,8,80,80]")
        skip = self.down1(latent)
        hidden = self.down2(skip)
        for block in self.convnext:
            hidden = block(hidden)
        hidden = F.gelu(hidden)
        hidden = torch.cat((self.up2d(hidden), skip), dim=1)
        hidden = self.fuse2d(hidden)
        volume = self.to_volume(hidden).reshape(len(latent), 128, 5, 40, 40)
        volume = self.up3d_1(volume)
        volume = self.up3d_2(volume)
        volume = self.up3d_3(volume)
        return self.mask_logits(volume), self.offset_raw(volume)


@dataclass
class LPCROutput:
    latent: torch.Tensor
    occupancy_logits: torch.Tensor
    offsets_raw: torch.Tensor


class R2LDMLidarLPCR(nn.Module):
    def __init__(self, grid: R2LDMGrid = R2LDMGrid(), *,
                 offset_activation: str = "legacy_sigmoid"):
        super().__init__()
        if offset_activation not in ("legacy_sigmoid", "centered_tanh"):
            raise ValueError("Unknown LPCR offset activation")
        self.grid = grid
        self.offset_activation = offset_activation
        self.encoder = R2LDMSparseLidarEncoder(grid)
        self.decoder = R2LDMPointReconstructionDecoder()

    def forward(self, means: torch.Tensor, coords: torch.Tensor,
                batch_size: int) -> LPCROutput:
        # Stage 0 bounds the sparse LiDAR embedding before S2D decoding.
        latent = torch.tanh(self.encoder(means, coords, batch_size))
        logits, offsets = self.decoder(latent)
        return LPCROutput(latent, logits, offsets)

    def activate_offsets(self, raw: torch.Tensor) -> torch.Tensor:
        if self.offset_activation == "legacy_sigmoid":
            # R2LDM's released branch, retained for an exact historical ablation.
            return torch.sigmoid(raw) * (51.2 / self.grid.shape_zyx[2])
        half_cell = raw.new_tensor(self.grid.voxel_xyz) * 0.5
        return torch.tanh(raw) * half_cell

    @torch.no_grad()
    def reconstruct(self, output: LPCROutput, threshold: float = 0.5) -> list[torch.Tensor]:
        if not 0 < threshold < 1:
            raise ValueError("LPCR occupancy threshold must lie in (0,1)")
        occupied = torch.softmax(output.occupancy_logits, dim=1)[:, 0] >= threshold
        indices = occupied.nonzero(as_tuple=False)
        points = []
        for batch_index in range(len(output.latent)):
            selected = indices[indices[:, 0] == batch_index]
            if not len(selected):
                points.append(output.offsets_raw.new_empty((0, 3)))
                continue
            raw = output.offsets_raw.permute(0, 2, 3, 4, 1)[
                selected[:, 0], selected[:, 1], selected[:, 2], selected[:, 3]]
            xyz_index = selected[:, [3, 2, 1]].to(raw.dtype)
            center = (raw.new_tensor(self.grid.minimum_xyz) +
                      (xyz_index + 0.5) * raw.new_tensor(self.grid.voxel_xyz))
            points.append(center + self.activate_offsets(raw))
        return points


def lpcr_loss(model: R2LDMLidarLPCR, output: LPCROutput,
              coords: torch.Tensor, means: torch.Tensor,
              offset_weight: float = 0.04) -> dict[str, torch.Tensor]:
    """Stage-0 fine-voxel occupancy CE and occupied-cell XYZ offset L1."""
    if offset_weight < 0 or coords.shape != (len(means), 4) or means.shape[1] != 3:
        raise ValueError("Invalid LPCR targets or offset weight")
    logits = output.occupancy_logits
    expected = (model.grid.shape_zyx)
    if logits.shape[1:] != (2, *expected) or output.offsets_raw.shape != (
            logits.shape[0], 3, *expected):
        raise ValueError("LPCR decoder output does not match its grid")
    targets = torch.ones((logits.shape[0], *expected), dtype=torch.long,
                         device=logits.device)
    b, z, y, x = coords.long().unbind(dim=1)
    targets[b, z, y, x] = 0  # R2LDM class zero means occupied.
    mask = F.cross_entropy(logits, targets)
    raw = output.offsets_raw.permute(0, 2, 3, 4, 1)[b, z, y, x]
    xyz_index = coords[:, [3, 2, 1]].to(means.dtype)
    center = (means.new_tensor(model.grid.minimum_xyz) +
              (xyz_index + 0.5) * means.new_tensor(model.grid.voxel_xyz))
    offset = F.l1_loss(model.activate_offsets(raw), means - center)
    total = mask + offset_weight * offset
    with torch.no_grad():
        predicted = logits.argmax(dim=1) == 0
        actual = targets == 0
        tp = (predicted & actual).sum()
        union = (predicted | actual).sum()
        iou = tp.float() / union.clamp_min(1)
    return {"loss": total, "occupancy": mask, "offset": offset, "iou": iou}

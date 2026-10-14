"""Frozen wrapper around the unmodified Downloads/encoder.py PFE_Engine.

The original forward densifies its output; this wrapper calls its original
voxelization, input MLP and sparse backbone directly and retains the sparse
output. It never substitutes the separately developed Stage-I PFE adapter.
"""

from __future__ import annotations

import importlib.util
from functools import lru_cache
from pathlib import Path
import sys

import torch
from torch import nn

from models.radar_lidar_stage1.config import VoxelGrid
from models.radar_lidar_stage1.sparse import SparseSites, encode_keys


VENDOR = Path(__file__).parent / "vendor"
GRID = VoxelGrid((-51.2, -51.2, -2.0), (51.2, 51.2, 2.0), (0.2, 0.2, 0.25))
BOTTLENECK_CHANNELS = 64


def masked_bottleneck_sites(coords: torch.Tensor, rate: float, salt: int) -> torch.Tensor:
    """Select whole stride-8 cells, including every fine voxel they contain."""
    if not 0 <= rate < 1:
        raise ValueError("Mask rate must be in [0,1)")
    if not len(coords) or rate == 0:
        return torch.zeros(len(coords), dtype=torch.bool, device=coords.device)
    coarse = coords.long().clone()
    coarse[:, 1:] //= 8
    keys = encode_keys(coarse, GRID.scale_shape(8))
    hashed = (keys * 1103515245 + int(salt) * 12345) % 2147483647
    return hashed.float() / 2147483647 < rate


@lru_cache(maxsize=1)
def _load_vendor():
    def load(name, path):
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    config = load("crossmodal_original_pfe_config", VENDOR / "config.py")
    if (tuple(config.POINT_CLOUD_RANGE) != (-51.2, -51.2, -2.0, 51.2, 51.2, 2.0)
            or tuple(config.VOXEL_SIZE) != GRID.size_xyz
            or config.LATENT_DIM != BOTTLENECK_CHANNELS):
        raise ValueError("Unmodified Downloads PFE config differs from the versioned interface")
    # The supplied source uses absolute `import config`. Bind its own untouched
    # config only while executing the module; do not change the source file.
    previous = sys.modules.get("config")
    sys.modules["config"] = config
    try:
        try:
            module = load("crossmodal_original_pfe_encoder", VENDOR / "encoder.py")
        except AttributeError as exc:
            raise RuntimeError("The original Downloads encoder requires spconv and CUDA") from exc
    finally:
        if previous is None:
            sys.modules.pop("config", None)
        else:
            sys.modules["config"] = previous
    if module.spconv is None:
        raise RuntimeError("The original Downloads encoder requires spconv and CUDA")
    return module


def _extract_state(path: str | Path) -> dict:
    saved = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(saved, dict):
        for key in ("state_dict", "model_state_dict", "model"):
            if key in saved and isinstance(saved[key], dict):
                return saved[key]
    if not isinstance(saved, dict) or not saved or not all(isinstance(v, torch.Tensor) for v in saved.values()):
        raise ValueError("Expected a checkpoint containing PFE_Engine weights")
    return saved


class OriginalPFE(nn.Module):
    def __init__(self, checkpoint: str | Path | None = None, *, frozen: bool = False):
        super().__init__()
        if frozen and checkpoint is None:
            raise ValueError("Frozen original PFE requires an explicit pretrained checkpoint")
        if not torch.cuda.is_available():
            raise RuntimeError("The unmodified original PFE_Engine constructor requires CUDA")
        vendor = _load_vendor()
        self.encoder = vendor.PFE_Engine(num_input_features=4, latent_dim=BOTTLENECK_CHANNELS)
        if checkpoint is not None:
            checkpoint = Path(checkpoint)
            if not checkpoint.is_file():
                raise FileNotFoundError(f"Pretrained original PFE weights missing: {checkpoint}")
            self.encoder.load_state_dict(_extract_state(checkpoint), strict=True)
        self.source_checkpoint = str(checkpoint.resolve()) if checkpoint is not None else None
        self.frozen = frozen
        if frozen:
            self.encoder.eval()
            self.encoder.requires_grad_(False)

    def train(self, mode: bool = True):
        super().train(False if self.frozen else mode)
        return self

    def forward(self, points: torch.Tensor, valid: torch.Tensor, *,
                input_mask_rate: float = 0.0, mask_salt: int = 0) -> SparseSites:
        if self.frozen:
            if input_mask_rate:
                raise ValueError("Frozen pretrained encoder cannot mask its input")
            with torch.no_grad():
                return self._encode(points, valid, input_mask_rate, mask_salt)
        return self._encode(points, valid, input_mask_rate, mask_salt)

    def _encode(self, points: torch.Tensor, valid: torch.Tensor,
                input_mask_rate: float, mask_salt: int) -> SparseSites:
        if points.ndim != 3 or valid.shape != points.shape[:2]:
            raise ValueError("Expected padded points and validity mask")
        point_list = [points[b, valid[b], :3].contiguous() for b in range(len(points))]
        features, coords = self.encoder._voxelize_batch(point_list)
        shape = GRID.scale_shape(8)
        if features is None:
            return SparseSites._from_validated(
                points.new_empty((0, 4), dtype=torch.long),
                points.new_empty((0, BOTTLENECK_CHANNELS)), shape, 8)
        if input_mask_rate:
            mask = masked_bottleneck_sites(coords, input_mask_rate, mask_salt)
            features = features.masked_fill(mask[:, None], 0.)
        x = self.encoder.input_mlp(features)
        sparse = _load_vendor().spconv.SparseConvTensor(x, coords.int(),
                                         self.encoder.grid_size_zyx, len(points))
        output = self.encoder.backbone(sparse)
        coordinates = output.indices.long()
        order = torch.argsort(encode_keys(coordinates, shape), stable=True)
        return SparseSites._from_validated(coordinates[order], output.features[order], shape, 8)


class FrozenOriginalPFE(OriginalPFE):
    def __init__(self, checkpoint: str | Path):
        super().__init__(checkpoint, frozen=True)

"""Stage-I pruning over an exact three-radar-voxel, three-metre domain.

Radar support is a hard coordinate constraint. The network may only remove
candidate cells; neither training nor inference can introduce a cell outside
the support mask. The faulty LiDAR is context and its observed cells are never
emitted as reconstruction additions.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import math

import numpy as np
from scipy.spatial import cKDTree
import torch
from torch import nn
from torch.nn import functional as F

from models.radar_lidar_crossmodal_v1.encoder import GRID as RADAR_GRID
from models.radar_lidar_stage2.sparse_geometry import GRID as GRID


VERSION = "radar3vox_3m_pruner_v1"
VERSION_1P6M = "radar3vox_1p6m_pruner_v1"
PRUNER_VERSIONS = (VERSION, VERSION_1P6M)
SUPPORT_RADIUS_M = 3.0
SUPPORT_MIN_VOXELS = 3
SHAPE = GRID.shape_zyx
CELL_COUNT = int(np.prod(SHAPE))


def _all_centers() -> np.ndarray:
    flat = np.arange(CELL_COUNT, dtype=np.int64)
    yz = SHAPE[1] * SHAPE[2]
    z, rem = divmod(flat, yz)
    y, x = divmod(rem, SHAPE[2])
    index = np.stack((x, y, z), axis=1)
    return (np.asarray(GRID.minimum_xyz, dtype=np.float64)
            + (index + 0.5) * np.asarray(GRID.voxel_xyz, dtype=np.float64))


_CENTERS = _all_centers()


def unique_radar_voxel_centers(radar_xyz: torch.Tensor | np.ndarray) -> np.ndarray:
    """Deduplicate the aligned radar returns on the global fine voxel grid.

    The global XY grid and expanded Z grid include radar just outside Stage
    II's ROI, so boundary candidates receive their proper 3 m support.
    """
    xyz = (radar_xyz.detach().cpu().numpy() if isinstance(radar_xyz, torch.Tensor)
           else np.asarray(radar_xyz))
    if xyz.ndim != 2 or xyz.shape[1] < 3:
        raise ValueError("Radar must have XYZ columns")
    xyz = xyz[:, :3]
    if not np.isfinite(xyz).all():
        raise ValueError("Radar XYZ contains nonfinite values")
    low = np.asarray((RADAR_GRID.minimum_xyz[0], RADAR_GRID.minimum_xyz[1], -8.0))
    high = np.asarray((RADAR_GRID.maximum_xyz[0], RADAR_GRID.maximum_xyz[1], 8.0))
    size = np.asarray(RADAR_GRID.size_xyz)
    inside = ((xyz >= low) & (xyz < high)).all(axis=1)
    if not inside.any():
        return np.empty((0, 3), dtype=np.float64)
    indices = np.unique(np.floor((xyz[inside] - low) / size).astype(np.int32), axis=0)
    return low + (indices.astype(np.float64) + 0.5) * size


def version_for_radius(radius_m: float) -> str:
    if radius_m == 3.0:
        return VERSION
    if radius_m == 1.6:
        return VERSION_1P6M
    raise ValueError("Pruner radius must be exactly 1.6 m or 3.0 m")


def radar_support_mask(radar_xyz: torch.Tensor | np.ndarray,
                       radius_m: float = SUPPORT_RADIUS_M,
                       min_radar_voxels: int = SUPPORT_MIN_VOXELS) -> np.ndarray:
    """True where enough distinct radar voxel centres lie within radius_m."""
    if not math.isfinite(radius_m) or radius_m <= 0:
        raise ValueError("Radar support radius must be positive and finite")
    if isinstance(min_radar_voxels, bool) or not isinstance(min_radar_voxels, int) or min_radar_voxels < 1:
        raise ValueError("Minimum distinct radar voxels must be a positive integer")
    radar_centers = unique_radar_voxel_centers(radar_xyz)
    if len(radar_centers) < min_radar_voxels:
        return np.zeros(SHAPE, dtype=np.bool_)
    tree = cKDTree(radar_centers)
    mask = np.empty(CELL_COUNT, dtype=np.bool_)
    for start in range(0, CELL_COUNT, 65536):
        stop = min(start + 65536, CELL_COUNT)
        distance, _ = tree.query(_CENTERS[start:stop], k=min_radar_voxels,
                                 distance_upper_bound=radius_m + 1e-9,
                                 workers=1)
        kth = distance if min_radar_voxels == 1 else distance[:, -1]
        mask[start:stop] = kth <= radius_m + 1e-9
    return mask.reshape(SHAPE)


def supported_coordinates(mask: np.ndarray | torch.Tensor,
                          device: torch.device | str = "cpu") -> torch.Tensor:
    """Sorted local [z,y,x] coordinates; no coordinates outside the mask."""
    values = np.asarray(mask.detach().cpu().numpy() if isinstance(mask, torch.Tensor)
                        else mask, dtype=np.bool_)
    if values.shape != SHAPE:
        raise ValueError(f"Support mask must have shape {SHAPE}")
    flat = np.flatnonzero(values.ravel())
    z, rem = divmod(flat, SHAPE[1] * SHAPE[2])
    y, x = divmod(rem, SHAPE[2])
    return torch.as_tensor(np.stack((z, y, x), axis=1).copy(),
                           dtype=torch.long, device=device)


def _grid_flat(xyz: torch.Tensor) -> torch.Tensor:
    minimum = xyz.new_tensor(GRID.minimum_xyz)
    maximum = xyz.new_tensor(GRID.maximum_xyz)
    size = xyz.new_tensor(GRID.voxel_xyz)
    inside = ((xyz >= minimum) & (xyz < maximum)).all(dim=1)
    index = torch.floor((xyz[inside] - minimum) / size).long()
    flat = (index[:, 2] * SHAPE[1] + index[:, 1]) * SHAPE[2] + index[:, 0]
    return flat, inside


def _input_volume(radar: torch.Tensor, faulty: torch.Tensor) -> torch.Tensor:
    if radar.ndim != 2 or radar.shape[1] != 7 or faulty.ndim != 2 or faulty.shape[1] != 3:
        raise ValueError("Expected radar [N,7] and faulty LiDAR [M,3]")
    if radar.device != faulty.device:
        raise ValueError("Radar and faulty LiDAR must share a device")
    flat, inside = _grid_flat(radar[:, :3])
    radar_inside = radar[inside]
    count = radar.new_zeros(CELL_COUNT).index_add_(
        0, flat, radar.new_ones(len(flat)))
    channels = [count.log1p()]
    for column, scale in ((3, 30.0), (5, 15.0), (6, 19.0)):
        value = (radar_inside[:, column] / scale).tanh()
        summed = radar.new_zeros(CELL_COUNT).index_add_(0, flat, value)
        channels.append(summed / count.clamp_min(1))
    lidar_flat, _ = _grid_flat(faulty)
    lidar_count = faulty.new_zeros(CELL_COUNT).index_add_(
        0, lidar_flat, faulty.new_ones(len(lidar_flat)))
    channels.append(lidar_count.log1p())
    local = F.avg_pool3d(lidar_count.reshape(1, 1, *SHAPE),
                         kernel_size=5, stride=1, padding=2).reshape(-1)
    channels.append(local)
    return torch.stack(channels).reshape(1, 6, *SHAPE)


def _block(inputs: int, outputs: int, *, stride: int = 1) -> nn.Sequential:
    return nn.Sequential(nn.Conv3d(inputs, outputs, 3, stride=stride, padding=1),
                         nn.GroupNorm(4, outputs), nn.SiLU())


@dataclass(frozen=True)
class RadarPrunerConfig:
    channels: int = 16
    negative_loss_weight: float = 0.35
    support_radius_m: float = SUPPORT_RADIUS_M
    dropout_p: float = 0.0

    def __post_init__(self) -> None:
        if (self.channels < 8 or self.channels % 4 or self.negative_loss_weight < 0
                or not math.isfinite(self.dropout_p) or not 0 <= self.dropout_p < 1):
            raise ValueError("Invalid radar pruner configuration")
        version_for_radius(self.support_radius_m)


class RadarPruner(nn.Module):
    """Small dense context network; scores every radar-supported fine cell."""

    def __init__(self, config: RadarPrunerConfig | None = None):
        super().__init__()
        self.config = config or RadarPrunerConfig()
        c = self.config.channels
        self.stem = _block(6, c)
        self.down1 = _block(c, 2 * c, stride=2)
        self.down2 = _block(2 * c, 4 * c, stride=2)
        self.bottleneck_dropout = nn.Dropout3d(self.config.dropout_p)
        self.up1 = _block(6 * c, 2 * c)
        self.up0 = _block(3 * c, c)
        self.score = nn.Conv3d(c, 1, 1)
        self.conditioning = nn.Linear(4 * c, 64)

    def forward(self, radar: torch.Tensor, faulty: torch.Tensor,
                support: np.ndarray | torch.Tensor) -> dict:
        coords = supported_coordinates(support, device=radar.device)
        observed_flat, _ = _grid_flat(faulty)
        flat = (coords[:, 0] * SHAPE[1] + coords[:, 1]) * SHAPE[2] + coords[:, 2]
        if len(observed_flat):
            coords = coords[~torch.isin(flat, torch.unique(observed_flat))]
        x0 = self.stem(_input_volume(radar, faulty))
        x1 = self.down1(x0)
        x2 = self.bottleneck_dropout(self.down2(x1))
        x = self.up1(torch.cat((F.interpolate(x2, size=x1.shape[-3:],
                                             mode="trilinear", align_corners=False), x1), 1))
        x = self.up0(torch.cat((F.interpolate(x, size=x0.shape[-3:],
                                            mode="trilinear", align_corners=False), x0), 1))
        logits = self.score(x)[0, 0, coords[:, 0], coords[:, 1], coords[:, 2]]
        coarse = F.avg_pool3d(x2, kernel_size=2, stride=2)[0]
        source = self.conditioning(coarse.permute(1, 2, 3, 0).reshape(-1, 4 * self.config.channels))
        zz, yy, xx = torch.meshgrid(
            torch.arange(2, device=radar.device),
            torch.arange(20, device=radar.device),
            torch.arange(20, device=radar.device), indexing="ij")
        source_coords = torch.stack((zz.reshape(-1), yy.reshape(-1) + 22,
                                     xx.reshape(-1) + 32), dim=1)
        source_coords = torch.cat((torch.zeros((len(source_coords), 1),
                                                dtype=torch.long, device=radar.device),
                                   source_coords), dim=1)
        return {"coords": coords, "logits": logits,
                "conditioning_coords": source_coords,
                "conditioning_features": source}

    def exported(self, output: dict) -> dict:
        coords = output["coords"]
        global_coords = coords + coords.new_tensor((0, 176, 256))
        global_coords = torch.cat((global_coords.new_zeros((len(coords), 1)),
                                   global_coords), dim=1)
        return {"confidence_coords": global_coords,
                "confidence_probabilities": output["logits"].sigmoid()[:, None],
                "conditioning_coords": output["conditioning_coords"],
                "conditioning_features": output["conditioning_features"],
                "grid_metadata": {"origin_xyz_m": RADAR_GRID.minimum_xyz,
                                  "voxel_xyz_m": RADAR_GRID.size_xyz,
                                  "conditioning_stride": 8,
                                  "coordinate_order": "batch,z,y,x"},
                "interface_version": version_for_radius(self.config.support_radius_m)}


def checkpoint_payload(model: RadarPruner, **metadata) -> dict:
    return {"version": version_for_radius(model.config.support_radius_m),
            "config": asdict(model.config),
            "support_radius_m": model.config.support_radius_m,
            "support_min_distinct_radar_voxels": SUPPORT_MIN_VOXELS,
            "model": model.state_dict(), **metadata}


def load_radar_pruner(path: str | Path, device: str = "cuda") -> tuple[RadarPruner, dict]:
    saved = torch.load(path, map_location="cpu", weights_only=False)
    radius_m = saved.get("support_radius_m")
    if (radius_m not in (1.6, 3.0) or
            saved.get("version") != version_for_radius(radius_m) or
            saved.get("support_min_distinct_radar_voxels") != SUPPORT_MIN_VOXELS):
        raise ValueError("Checkpoint has another radar-support policy")
    model = RadarPruner(RadarPrunerConfig(**saved["config"]))
    if model.config.support_radius_m != radius_m:
        raise ValueError("Pruner model and checkpoint radii differ")
    model.load_state_dict(saved["model"], strict=True)
    return model.to(device).eval(), saved

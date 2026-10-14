"""Four-scale Stage-I adapter for the PFE_Engine in Downloads/encoder.py.

The supplied PFE backbone is kept sparse and exposed at every scale. Stage I
cannot consume its final dense tensor: correspondence and surface proposals
need S1, S2, S3 and S4 sites. Radar and LiDAR use independent copies.
"""

from __future__ import annotations

import torch
from torch import nn

from .config import VoxelGrid
from .sparse import SparseSites, axial_connectivity, encode_keys, voxelize

try:
    import spconv.pytorch as spconv
except ImportError:
    spconv = None


def require_spconv() -> None:
    if spconv is None:
        raise RuntimeError(
            "Stage I encoder_backend=pfe_spconv requires spconv in this Python "
            "environment. Install a spconv build compatible with its CUDA/PyTorch "
            "runtime, or use the native Stage I config."
        )


class PFEPointVoxelEncoder(nn.Module):
    """PFE mean-XYZ/log-density MLP plus modality-specific measured attributes."""

    def __init__(self, grid: VoxelGrid, modality: str):
        super().__init__()
        if modality not in ("radar", "lidar"):
            raise ValueError(modality)
        self.grid, self.modality = grid, modality
        self.raw_columns = 7 if modality == "radar" else 4
        attribute_count = self.raw_columns - 3
        # This is the 4 -> 16 -> 32 voxel MLP from the supplied encoder.
        self.input_mlp = nn.Sequential(
            nn.Linear(4, 16), nn.BatchNorm1d(16), nn.ReLU(),
            nn.Linear(16, 32), nn.BatchNorm1d(32), nn.ReLU())
        self.attribute_mlp = nn.Sequential(
            nn.Linear(attribute_count * 2, 16), nn.LayerNorm(16), nn.ReLU(),
            nn.Linear(16, 32))

    def forward(self, points: torch.Tensor, valid: torch.Tensor, *,
                defer_diagnostics: bool = False, check_finite: bool = True):
        if points.shape[-1] != self.raw_columns:
            raise ValueError(f"{self.modality} expects {self.raw_columns} point columns")
        coords, inverse, selected, inside = voxelize(
            points, valid, self.grid, check_finite=check_finite)
        n = len(coords)
        if not n:
            empty = SparseSites._from_validated(
                coords, points.new_empty((0, 32)), self.grid.shape_zyx)
            return empty, {"points": 0, "active": 0, "mean_points": 0.,
                           "retained_fraction": 0.}
        counts = selected.new_zeros(n).index_add(
            0, inverse, selected.new_ones(len(selected)))
        mean_xyz = selected.new_zeros((n, 3)).index_add(
            0, inverse, selected[:, :3]) / counts[:, None].clamp_min(1)
        core = torch.cat((mean_xyz, torch.log(counts.clamp_min(1))[:, None]), dim=1)
        features = self.input_mlp(core)

        attrs = selected[:, 3:]
        if self.modality == "radar":
            attrs = torch.stack((torch.tanh(attrs[:, 0] / 30),
                                 torch.tanh(attrs[:, 1] / 15),
                                 torch.tanh(attrs[:, 2] / 15),
                                 attrs[:, 3] / 19), dim=-1)
        else:
            attrs = torch.sign(attrs) * torch.log1p(attrs.abs()) / 5
        mean_attrs = attrs.new_zeros((n, attrs.shape[1])).index_add(
            0, inverse, attrs) / counts[:, None].clamp_min(1)
        max_attrs = attrs.new_full((n, attrs.shape[1]), -torch.inf).scatter_reduce(
            0, inverse[:, None].expand_as(attrs), attrs, reduce="amax",
            include_self=True)
        features = features + self.attribute_mlp(torch.cat(
            (mean_attrs, max_attrs), dim=1))
        points_count = inside.sum()
        mean_points = counts.mean()
        retained_fraction = points_count / valid.sum().clamp_min(1)
        if not defer_diagnostics:
            points_count = int(points_count)
            mean_points = float(mean_points)
            retained_fraction = float(retained_fraction)
        return SparseSites._from_validated(coords, features, self.grid.shape_zyx), {
            "points": points_count, "active": n, "mean_points": mean_points,
            "retained_fraction": retained_fraction}


class SparseResBlock(nn.Module):
    """The two-submanifold-convolution residual block from encoder.py."""

    def __init__(self, channels: int, indice_key: str):
        super().__init__()
        require_spconv()
        self.conv1 = spconv.SubMConv3d(channels, channels, 3, padding=1,
                                       bias=False, indice_key=indice_key)
        self.bn1 = nn.BatchNorm1d(channels)
        self.relu = nn.ReLU()
        self.conv2 = spconv.SubMConv3d(channels, channels, 3, padding=1,
                                       bias=False, indice_key=indice_key)
        self.bn2 = nn.BatchNorm1d(channels)

    def forward(self, sites):
        residual = sites.features
        out = self.conv1(sites)
        out = out.replace_feature(self.relu(self.bn1(out.features)))
        out = self.conv2(out)
        return out.replace_feature(self.relu(self.bn2(out.features) + residual))


class PFESpconvBackbone(nn.Module):
    """PFE_Engine's 1, 1/2, 1/4 and 1/8 stages without densification.

    PFEPointVoxelEncoder supplies 32 channels to the first sparse layer and
    retains RCS, Doppler, scan age and LiDAR reflectivity.
    """

    def __init__(self, channels: tuple[int, int, int, int]):
        super().__init__()
        require_spconv()
        if len(channels) != 4:
            raise ValueError("PFE encoder requires four output scales")
        c1, c2, c3, c4 = channels
        self.channels = channels
        self.stem = spconv.SubMConv3d(32, c1, 3, padding=1, bias=False,
                                      indice_key="subm0")
        self.stem_bn = nn.BatchNorm1d(c1)
        self.res1 = SparseResBlock(c1, "res_scale1a")
        self.down1 = spconv.SparseConv3d(c1, c2, 3, stride=2, padding=1,
                                         bias=False, indice_key="down1")
        self.down1_bn = nn.BatchNorm1d(c2)
        self.res2 = SparseResBlock(c2, "res_scale2a")
        self.down2 = spconv.SparseConv3d(c2, c3, 3, stride=2, padding=1,
                                         bias=False, indice_key="down2")
        self.down2_bn = nn.BatchNorm1d(c3)
        self.res3a = SparseResBlock(c3, "res_scale4a")
        self.res3b = SparseResBlock(c3, "res_scale4b")
        self.down3 = spconv.SparseConv3d(c3, c4, 3, stride=2, padding=1,
                                         bias=False, indice_key="down3")
        self.down3_bn = nn.BatchNorm1d(c4)
        self.res4a = SparseResBlock(c4, "res_scale8a")
        self.final = spconv.SubMConv3d(c4, c4, 1, bias=False,
                                       indice_key="final_compress")
        self.final_bn = nn.BatchNorm1d(c4)
        self.res4b = SparseResBlock(c4, "res_latent")
        self.relu = nn.ReLU()

    @staticmethod
    def _as_sites(tensor, shape_zyx: tuple[int, int, int], stride: int) -> SparseSites:
        coords = tensor.indices.long()
        order = torch.argsort(encode_keys(coords, shape_zyx), stable=True)
        return SparseSites._from_validated(coords[order], tensor.features[order],
                                           shape_zyx, stride)

    def forward(self, sites: SparseSites, *, defer_diagnostics: bool = False):
        require_spconv()
        shape = sites.shape_zyx
        if any(n % 8 for n in shape):
            raise ValueError("PFE downsampling requires grid dimensions divisible by eight")
        if not len(sites.coords):
            levels = [SparseSites._from_validated(
                sites.coords.new_empty((0, 4)),
                sites.features.new_empty((0, channels)),
                tuple(n // (2 ** index) for n in shape), 2 ** index)
                for index, channels in enumerate(self.channels)]
            return levels, [self._stats(level, 0, 0, defer_diagnostics)
                            for level in levels]

        batch_size = int(sites.coords[:, 0].max()) + 1
        x = spconv.SparseConvTensor(sites.features.contiguous(), sites.coords.int().contiguous(),
                                    list(shape), batch_size)
        levels, stats = [], []
        x = self.stem(x)
        x = x.replace_feature(self.relu(self.stem_bn(x.features)))
        x = self.res1(x)
        levels.append(self._as_sites(x, shape, 1))
        stats.append(self._stats(levels[-1], len(sites.coords), len(sites.coords), defer_diagnostics))

        for index, (down, bn, blocks) in enumerate((
            (self.down1, self.down1_bn, (self.res2,)),
            (self.down2, self.down2_bn, (self.res3a, self.res3b)),
            (self.down3, self.down3_bn, (self.res4a,))), start=1):
            before = len(x.indices)
            x = down(x)
            x = x.replace_feature(self.relu(bn(x.features)))
            after = len(x.indices)
            for block in blocks:
                x = block(x)
            if index == 3:
                x = self.final(x)
                x = x.replace_feature(self.relu(self.final_bn(x.features)))
                x = self.res4b(x)
            stride = 2 ** index
            levels.append(self._as_sites(x, tuple(n // stride for n in shape), stride))
            stats.append(self._stats(levels[-1], before, after, defer_diagnostics))
        return levels, stats

    @staticmethod
    def _stats(level: SparseSites, pre_down: int, post_down: int,
               defer_diagnostics: bool) -> dict:
        return {"pre_down": pre_down, "post_down": post_down,
                "post_growth": post_down, "post_block": len(level.coords),
                "stride": level.stride, "channels": level.features.shape[-1],
                **axial_connectivity(level, defer_diagnostics=defer_diagnostics)}

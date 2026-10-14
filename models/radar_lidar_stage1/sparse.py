"""True coordinate-indexed 3D sparse convolutions in portable PyTorch.

No dense 3D volume is allocated. Learned 3×3×3 kernels gather only active
neighbors. This reference backend is correct but slower than optimized spconv.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import product

import torch
from torch import nn
from torch.nn import functional as F

from .config import VoxelGrid


def encode_keys(coords: torch.Tensor, shape_zyx: tuple[int, int, int]) -> torch.Tensor:
    z, y, x = shape_zyx
    if coords.ndim != 2 or coords.shape[-1] != 4:
        raise ValueError("Sparse coordinates must be [N, batch,z,y,x]")
    if coords.numel() and ((coords < 0).any() or (coords[:, 1:] >= coords.new_tensor((z, y, x))).any()):
        raise ValueError("Sparse coordinates outside physical grid")
    return _encode_keys_unchecked(coords,shape_zyx)


def _encode_keys_unchecked(coords: torch.Tensor, shape_zyx: tuple[int, int, int]) -> torch.Tensor:
    """Only for coordinates already bounded by voxelization or target construction."""
    z,y,x=shape_zyx
    return (((coords[:, 0] * z + coords[:, 1]) * y + coords[:, 2]) * x + coords[:, 3]).long()


def decode_keys(keys: torch.Tensor, shape_zyx: tuple[int, int, int]) -> torch.Tensor:
    z, y, x = shape_zyx
    rest, xx = torch.div(keys, x, rounding_mode="floor"), keys % x
    rest, yy = torch.div(rest, y, rounding_mode="floor"), rest % y
    bb, zz = torch.div(rest, z, rounding_mode="floor"), rest % z
    return torch.stack((bb, zz, yy, xx), dim=1)


@dataclass
class SparseSites:
    coords: torch.Tensor  # [N,4], sorted unique BZYX
    features: torch.Tensor  # [N,C]
    shape_zyx: tuple[int, int, int]
    stride: int = 1
    # Integer neighbor indices depend on coordinates, not learned features.
    # replace_features keeps this cache across submanifold convolutions.
    _neighbor_cache: dict = field(default_factory=dict, repr=False, compare=False)
    # Internal constructors have already produced bounded, sorted unique keys.
    # Rechecking them on every feature replacement would synchronize the GPU.
    _validated_coordinates: bool = field(default=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self.coords.shape != (len(self.features), 4):
            raise ValueError("Coordinate and feature counts differ")
        if not self._validated_coordinates:
            keys = encode_keys(self.coords, self.shape_zyx)
            if len(keys) > 1 and not bool(torch.all(keys[1:] > keys[:-1])):
                raise ValueError("Sparse coordinates must be sorted and unique")

    @classmethod
    def _from_validated(cls, coords: torch.Tensor, features: torch.Tensor,
                        shape_zyx: tuple[int, int, int], stride: int = 1,
                        neighbor_cache: dict | None = None) -> "SparseSites":
        return cls(coords, features, shape_zyx, stride,
                   {} if neighbor_cache is None else neighbor_cache, True)

    @property
    def keys(self) -> torch.Tensor:
        return _encode_keys_unchecked(self.coords, self.shape_zyx)

    def centers_xyz(self, grid: VoxelGrid) -> torch.Tensor:
        return grid.centers_xyz(self.coords, self.stride)

    def replace_features(self, features: torch.Tensor) -> "SparseSites":
        return SparseSites._from_validated(self.coords, features, self.shape_zyx,
                                           self.stride, self._neighbor_cache)


def voxelize(points: torch.Tensor, valid: torch.Tensor, grid: VoxelGrid, *, check_finite: bool = True) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Map B×P×C points to sorted sparse cells; return coords, inverse, selected points, in-grid mask."""
    if points.ndim != 3 or valid.shape != points.shape[:2] or points.shape[-1] < 3:
        raise ValueError("Expected points [B,P,C] and valid [B,P]")
    if check_finite and not bool(torch.isfinite(points[valid]).all()):
        raise ValueError("Valid sensor points must be finite")
    minimum = points.new_tensor(grid.minimum_xyz)
    size = points.new_tensor(grid.size_xyz)
    xyz_idx = torch.floor((points[..., :3] - minimum) / size).long()
    shape_xyz = points.new_tensor(grid.shape_zyx[::-1], dtype=torch.long)
    inside = valid & ((xyz_idx >= 0) & (xyz_idx < shape_xyz)).all(-1)
    selected = points[inside]
    xyz_idx = xyz_idx[inside]
    batch_idx = torch.arange(points.shape[0], device=points.device)[:, None].expand_as(valid)[inside]
    rows = torch.stack((batch_idx, xyz_idx[:, 2], xyz_idx[:, 1], xyz_idx[:, 0]), dim=1)
    keys, inverse = torch.unique(_encode_keys_unchecked(rows, grid.shape_zyx), sorted=True, return_inverse=True)
    return decode_keys(keys, grid.shape_zyx), inverse, selected, inside


class PointVoxelEncoder(nn.Module):
    """Separate learned point embeddings and mean/max/count voxel aggregation."""

    def __init__(self, grid: VoxelGrid, modality: str, out_channels: int):
        super().__init__()
        if modality not in {"radar", "lidar"}:
            raise ValueError(modality)
        self.grid, self.modality = grid, modality
        self.raw_columns = 7 if modality == "radar" else 4
        input_dim = 3 + 3 + (4 if modality == "radar" else 1)
        self.point = nn.Sequential(nn.Linear(input_dim, out_channels), nn.LayerNorm(out_channels), nn.SiLU(), nn.Linear(out_channels, out_channels))
        self.aggregate = nn.Sequential(nn.Linear(out_channels * 2 + 1, out_channels), nn.LayerNorm(out_channels), nn.SiLU())

    def forward(self, points: torch.Tensor, valid: torch.Tensor, *, defer_diagnostics: bool = False,
                check_finite: bool = True) -> tuple[SparseSites, dict]:
        if points.shape[-1] != self.raw_columns:
            raise ValueError(f"{self.modality} expects {self.raw_columns} point columns")
        coords, inverse, selected, inside = voxelize(points, valid, self.grid, check_finite=check_finite)
        n = len(coords)
        if n == 0:
            return SparseSites._from_validated(coords, points.new_zeros((0, self.point[0].out_features)), self.grid.shape_zyx), {"points": 0, "active": 0, "mean_points": 0.0, "retained_fraction": 0.0}
        center = self.grid.centers_xyz(coords)[inverse]
        offset = (selected[:, :3] - center) / selected.new_tensor(self.grid.size_xyz)
        global_xyz = (selected[:, :3] - selected.new_tensor(self.grid.minimum_xyz)) / selected.new_tensor(tuple(h-l for l,h in zip(self.grid.minimum_xyz,self.grid.maximum_xyz)))
        attrs = selected[:, 3:]
        if self.modality == "radar":
            attrs = torch.stack((torch.tanh(attrs[:, 0]/30), torch.tanh(attrs[:, 1]/15), torch.tanh(attrs[:, 2]/15), attrs[:, 3]/19), dim=-1)
        else:
            attrs = torch.sign(attrs)*torch.log1p(attrs.abs())/5
        embeds = self.point(torch.cat((offset, global_xyz, attrs), dim=-1))
        mean = embeds.new_zeros((n, embeds.shape[-1])).index_add(0, inverse, embeds)
        counts = embeds.new_zeros(n).index_add(0, inverse, torch.ones_like(inverse, dtype=embeds.dtype))
        mean = mean / counts.clamp_min(1)[:, None]
        maximum = embeds.new_full((n, embeds.shape[-1]), -torch.inf).scatter_reduce(0, inverse[:, None].expand_as(embeds), embeds, reduce="amax", include_self=True)
        features = self.aggregate(torch.cat((mean, maximum, torch.log1p(counts)[:, None]), dim=-1))
        points_count = inside.sum()
        mean_points = counts.mean()
        retained_fraction = points_count / valid.sum().clamp_min(1)
        if not defer_diagnostics:
            points_count,mean_points,retained_fraction = int(points_count),float(mean_points),float(retained_fraction)
        return SparseSites._from_validated(coords, features, self.grid.shape_zyx), {"points": points_count, "active": n, "mean_points": mean_points, "retained_fraction": retained_fraction}


_OFFSETS = tuple(product((-1, 0, 1), repeat=3))
_MAP_CHUNK_SIZE = 32768


def _kernel_map(sites: SparseSites, target: torch.Tensor, stride: int) -> tuple[tuple[torch.Tensor, torch.Tensor], ...]:
    """Build 27 GPU neighbor pair lists with batched lookup and bounded scratch space.

    Each pair is (output row, input row). The map has no gradient and can be
    reused by layers whose input and output coordinate sets are unchanged.
    """
    groups_out: list[list[torch.Tensor]] = [[] for _ in _OFFSETS]
    groups_in: list[list[torch.Tensor]] = [[] for _ in _OFFSETS]
    if not len(target) or not len(sites.coords):
        empty = target.new_empty((0,), dtype=torch.long)
        return tuple((empty, empty) for _ in _OFFSETS)
    with torch.no_grad():
        source_keys = sites.keys
        offsets = target.new_tensor(_OFFSETS)
        shape = target.new_tensor(sites.shape_zyx)
        z_size, y_size, x_size = sites.shape_zyx
        for begin in range(0, len(target), _MAP_CHUNK_SIZE):
            end = min(begin + _MAP_CHUNK_SIZE, len(target))
            rows = target[begin:end]
            xyz = rows[:, None, 1:] * stride + offsets[None, :, :]
            inside = ((xyz >= 0) & (xyz < shape)).all(-1)
            keys = (((rows[:, None, 0] * z_size + xyz[..., 0]) * y_size + xyz[..., 1]) * x_size + xyz[..., 2]).long()
            positions = torch.searchsorted(source_keys, keys.reshape(-1)).reshape(keys.shape)
            safe = positions.clamp_max(len(source_keys) - 1)
            present = inside & (positions < len(source_keys)) & (source_keys[safe] == keys)
            # Transposition makes the nonzero rows offset-major. One host
            # synchronization obtains all 27 group sizes, instead of one or
            # two GPU-to-CPU decisions for every offset in every convolution.
            kernel_index, output_local = torch.where(present.T)
            input_index = positions.T[kernel_index, output_local]
            counts = torch.bincount(kernel_index, minlength=len(_OFFSETS)).tolist()
            for k, (out_rows, in_rows) in enumerate(zip(output_local.split(counts), input_index.split(counts))):
                groups_out[k].append(out_rows + begin)
                groups_in[k].append(in_rows)
    return tuple((torch.cat(groups_out[k]), torch.cat(groups_in[k])) for k in range(len(_OFFSETS)))


class SparseConv3d(nn.Module):
    """3×3×3 sparse convolution on active outputs; stride two coalesces sites."""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1, grow: bool = False):
        super().__init__()
        if stride not in (1, 2):
            raise ValueError("Only stride one/two are supported")
        self.stride, self.grow = stride, grow
        self.kernel = nn.Parameter(torch.empty(27, in_channels, out_channels))
        self.bias = nn.Parameter(torch.zeros(out_channels))
        nn.init.kaiming_uniform_(self.kernel.view(27*in_channels, out_channels).T, a=1)

    def _target(self, sites: SparseSites) -> tuple[torch.Tensor, tuple[int, int, int]]:
        if self.stride == 2:
            shape = tuple(n//2 for n in sites.shape_zyx)
            if any(n % 2 for n in sites.shape_zyx):
                raise ValueError("Stride two requires even spatial shape")
            parent = sites.coords.clone()
            parent[:, 1:] //= 2
            keys = torch.unique(_encode_keys_unchecked(parent, shape), sorted=True)
            target = decode_keys(keys, shape)
        else:
            shape = sites.shape_zyx
            if self.grow and len(sites.coords):
                # Axial support growth; the convolution still uses the full 3×3×3 kernel.
                axial = ((0,0,0), (1,0,0), (-1,0,0), (0,1,0), (0,-1,0), (0,0,1), (0,0,-1))
                expanded = torch.cat([sites.coords + sites.coords.new_tensor((0,*o)) for o in axial], dim=0)
                inbounds = (expanded[:, 1:] >= 0).all(-1) & (expanded[:, 1:] < expanded.new_tensor(shape)).all(-1)
                keys = torch.unique(_encode_keys_unchecked(expanded[inbounds], shape), sorted=True)
                target = decode_keys(keys, shape)
            else:
                target = sites.coords
        return target, shape

    def forward(self, sites: SparseSites) -> SparseSites:
        target, shape = self._target(sites)
        output = sites.features.new_zeros((len(target), self.kernel.shape[-1]))
        if not len(target):
            return SparseSites._from_validated(target, output, shape, sites.stride*self.stride)
        reusable = self.stride == 1 and not self.grow
        cache_key = ("conv3d_3x3x3", sites.coords._version)
        pairs = sites._neighbor_cache.get(cache_key) if reusable else None
        if pairs is None:
            pairs = _kernel_map(sites, target, self.stride)
            if reusable:
                sites._neighbor_cache.clear()
                sites._neighbor_cache[cache_key] = pairs
        for kernel_index, (out_index, in_index) in enumerate(pairs):
            if len(out_index):
                values = sites.features.index_select(0, in_index) @ self.kernel[kernel_index]
                output = output.index_add(0, out_index, values)
        return SparseSites._from_validated(target, output + self.bias, shape,
                                           sites.stride*self.stride,
                                           sites._neighbor_cache if reusable else None)

    def forward_reference(self, sites: SparseSites) -> SparseSites:
        """Original offset-wise implementation for numerical and gradient checks."""
        target, shape = self._target(sites)
        output = sites.features.new_zeros((len(target), self.kernel.shape[-1]))
        if not len(target):
            return SparseSites(target, output, shape, sites.stride*self.stride)
        source_keys = sites.keys
        for kernel_index, offset in enumerate(_OFFSETS):
            neighbor = target.clone()
            neighbor[:, 1:] = neighbor[:, 1:] * self.stride + neighbor.new_tensor(offset)
            inbounds = (neighbor[:, 1:] >= 0).all(-1) & (neighbor[:, 1:] < neighbor.new_tensor(sites.shape_zyx)).all(-1)
            if not bool(inbounds.any()):
                continue
            out_index = torch.where(inbounds)[0]
            neighbor_keys = encode_keys(neighbor[inbounds], sites.shape_zyx)
            positions = torch.searchsorted(source_keys, neighbor_keys)
            present = (positions < len(source_keys)) & (source_keys[positions.clamp(max=max(len(source_keys)-1,0))] == neighbor_keys)
            if bool(present.any()):
                output = output.index_add(0, out_index[present], sites.features[positions[present]] @ self.kernel[kernel_index])
        return SparseSites(target, output + self.bias, shape, sites.stride*self.stride)


class SparseResidual(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv1, self.norm1 = SparseConv3d(channels, channels), nn.LayerNorm(channels)
        self.conv2, self.norm2 = SparseConv3d(channels, channels), nn.LayerNorm(channels)

    def forward(self, sites: SparseSites) -> SparseSites:
        x = sites.replace_features(F.silu(self.norm1(self.conv1(sites).features)))
        x = self.norm2(self.conv2(x).features)
        return sites.replace_features(F.silu(sites.features + x))


def receptive_fields(scales: int, growth_scales: tuple[int, ...] = ()) -> tuple[int, ...]:
    """Theoretical fine-grid RF including enabled support-growth convolutions."""
    rf, jump = 3, 1
    result = []
    for level in range(scales):
        if level:
            rf += 2*jump
            jump *= 2
        if level+1 in growth_scales:
            rf += 2*jump
        rf += 4*jump
        result.append(rf)
    return tuple(result)


@torch.no_grad()
def axial_connectivity(sites: SparseSites, *, defer_diagnostics: bool = False) -> dict[str, float | torch.Tensor]:
    """Fraction of active sites isolated from six axial neighbors at this scale."""
    if not len(sites.coords):
        return {"isolated_fraction":1.0,"mean_axial_neighbors":0.0}
    keys=sites.keys
    count=torch.zeros(len(keys),device=keys.device,dtype=torch.float32)
    z_size,y_size,x_size=sites.shape_zyx
    for axis in (1,2,3):
        for sign in (-1,1):
            neighbor=sites.coords.clone()
            neighbor[:,axis]+=sign
            inside=(neighbor[:,axis]>=0)&(neighbor[:,axis]<sites.shape_zyx[axis-1])
            # Search all rows, including out-of-bounds candidates, and mask
            # them afterward. This avoids a data-dependent GPU-to-CPU branch.
            query=(((neighbor[:,0]*z_size+neighbor[:,1])*y_size+neighbor[:,2])*x_size+neighbor[:,3]).long()
            pos=torch.searchsorted(keys,query)
            hit=inside&(pos<len(keys))&(keys[pos.clamp(max=len(keys)-1)]==query)
            count+=hit.float()
    isolated=(count==0).float().mean()
    neighbors=count.mean()
    if not defer_diagnostics:
        isolated,neighbors=float(isolated),float(neighbors)
    return {"isolated_fraction":isolated,"mean_axial_neighbors":neighbors}


class SparseBackbone(nn.Module):
    def __init__(self, channels: tuple[int, ...], growth_scales: tuple[int, ...]):
        super().__init__()
        self.stem = SparseConv3d(channels[0], channels[0])
        self.down = nn.ModuleList(SparseConv3d(a,b,stride=2) for a,b in zip(channels[:-1],channels[1:]))
        self.grow = nn.ModuleDict({str(i):SparseConv3d(channels[i-1],channels[i-1],grow=True) for i in growth_scales if 1 <= i <= len(channels)})
        self.blocks = nn.ModuleList(SparseResidual(c) for c in channels)
        self.channels = channels

    def forward(self, sites: SparseSites, *, defer_diagnostics: bool = False) -> tuple[list[SparseSites], list[dict]]:
        x = sites.replace_features(F.silu(self.stem(sites).features))
        levels, stats = [], []
        for i, block in enumerate(self.blocks):
            pre_down = len(x.coords)
            if i:
                x = self.down[i-1](x)
                x = x.replace_features(F.silu(x.features))
            post_down = len(x.coords)
            if str(i+1) in self.grow:
                x = self.grow[str(i+1)](x)
                x = x.replace_features(F.silu(x.features))
            post_growth = len(x.coords)
            x = block(x)
            levels.append(x)
            stats.append({"pre_down": pre_down, "post_down": post_down, "post_growth": post_growth, "post_block": len(x.coords), "stride": x.stride, "channels": x.features.shape[-1], **axial_connectivity(x,defer_diagnostics=defer_diagnostics)})
        return levels, stats

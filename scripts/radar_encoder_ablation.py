"""Radar autoencoders with VoxelNet-style or PointNet++-style encoders.

The cited models are encoders for detection, not published radar autoencoders.
Both variants therefore use one shared anchor-expansion decoder and objective.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


def gather(points: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
    batch = torch.arange(points.shape[0], device=points.device)
    return points[batch.view(-1, *([1] * (index.ndim - 1))), index]


@torch.no_grad()
def farthest_points(xyz: torch.Tensor, count: int) -> torch.Tensor:
    count = min(count, xyz.shape[1])
    if xyz.is_cuda:
        try:
            from models.pointops.functions import pointops
        except ModuleNotFoundError:
            # Stage-II radar uses this repo's ``models`` namespace; load the
            # compiled upstream pointops as its own top-level package.
            from pointops.functions import pointops
        return pointops.furthestsampling(xyz.contiguous(), count).long()
    # CPU fallback for focused tests; production uses the compiled CUDA FPS.
    batch, length, _ = xyz.shape
    chosen = torch.zeros((batch, count), dtype=torch.long, device=xyz.device)
    distance = torch.full((batch, length), float("inf"), device=xyz.device)
    farthest = torch.zeros(batch, dtype=torch.long, device=xyz.device)
    rows = torch.arange(batch, device=xyz.device)
    for step in range(count):
        chosen[:, step] = farthest
        distance = torch.minimum(distance, (xyz - xyz[rows, farthest, None]).square().sum(-1))
        farthest = distance.argmax(-1)
    return chosen


class SetAbstraction(nn.Module):
    """PointNet++ sampled centers, radius grouping, and local PointNet pooling."""

    def __init__(self, in_channels: int, out_channels: int, radii: tuple[float, float],
                 neighbors: tuple[int, int], dropout_p: float = 0.0):
        super().__init__()
        self.radii, self.neighbors = radii, neighbors
        half = out_channels // 2
        self.mlps = nn.ModuleList([
            nn.Sequential(nn.Linear(in_channels + 3, half), nn.ReLU(),
                          nn.Linear(half, half), nn.ReLU(),
                          *([nn.Dropout(dropout_p)] if dropout_p else []))
            for _ in radii])

    def forward(self, source_xyz: torch.Tensor, source_features: torch.Tensor,
                centers: torch.Tensor) -> torch.Tensor:
        distance = torch.cdist(centers, source_xyz)
        outputs = []
        for radius, neighbors, mlp in zip(self.radii, self.neighbors, self.mlps):
            near_distance, near_index = distance.topk(min(neighbors, source_xyz.shape[1]),
                                                     largest=False)
            relative = gather(source_xyz, near_index) - centers[:, :, None]
            features = gather(source_features, near_index)
            transformed = mlp(torch.cat((relative / radius, features), -1))
            # Always retain the nearest return, even for isolated radar points.
            valid = (near_distance <= radius) | (
                torch.arange(near_index.shape[-1], device=near_index.device) == 0)
            transformed = transformed.masked_fill(~valid[..., None], -1e4)
            outputs.append(transformed.amax(2))
        return torch.cat(outputs, -1)


class PointNetPPEncoder(nn.Module):
    def __init__(self, width: int, attribute_channels: int = 2,
                 dropout_p: float = 0.0):
        super().__init__()
        self.sa1 = SetAbstraction(attribute_channels, width, (1.6, 3.2),
                                  (24, 48), dropout_p)
        self.sa2 = SetAbstraction(width, width * 2, (3.2, 6.4),
                                  (24, 48), dropout_p)
        self.propagate = nn.Sequential(
            nn.Linear(width * 3, width), nn.ReLU(), nn.Linear(width, width),
            *([nn.Dropout(dropout_p)] if dropout_p else []))

    def forward(self, xyz: torch.Tensor, attrs: torch.Tensor,
                anchor_index: torch.Tensor) -> torch.Tensor:
        centers = gather(xyz, anchor_index)
        local = self.sa1(xyz, attrs, centers)
        coarse_index = farthest_points(centers, max(1, centers.shape[1] // 4))
        coarse_xyz = gather(centers, coarse_index)
        coarse = self.sa2(centers, local, coarse_xyz)
        distance = torch.cdist(centers, coarse_xyz)
        near_distance, near_index = distance.topk(min(3, coarse_xyz.shape[1]),
                                                 largest=False)
        weight = 1 / near_distance.clamp_min(1e-5)
        weight = weight / weight.sum(-1, keepdim=True)
        interpolated = (gather(coarse, near_index) * weight[..., None]).sum(2)
        return self.propagate(torch.cat((local, interpolated), -1))


class VoxelFeatureLayer(nn.Module):
    """Stacked point MLP and voxel max aggregation from VoxelNet's VFE idea."""

    def __init__(self, in_channels: int, width: int):
        super().__init__()
        self.point = nn.Sequential(nn.Linear(in_channels, width), nn.ReLU())
        self.fuse = nn.Sequential(nn.Linear(width * 2, width), nn.ReLU())

    def forward(self, features: torch.Tensor, inverse: torch.Tensor,
                voxel_count: int) -> tuple[torch.Tensor, torch.Tensor]:
        point = self.point(features)
        pooled = torch.full((voxel_count, point.shape[-1]), -1e4,
                            device=point.device, dtype=point.dtype)
        pooled.scatter_reduce_(0, inverse[:, None].expand_as(point), point,
                               reduce="amax", include_self=True)
        enhanced = self.fuse(torch.cat((point, pooled[inverse]), -1))
        voxels = torch.full_like(pooled, -1e4)
        voxels.scatter_reduce_(0, inverse[:, None].expand_as(enhanced), enhanced,
                               reduce="amax", include_self=True)
        return enhanced, voxels


class VoxelNetEncoder(nn.Module):
    def __init__(self, width: int, attribute_channels: int = 2,
                 origin_xyz: tuple[float, float, float] = (0.0, -25.6, -3.0)):
        super().__init__()
        import MinkowskiEngine as ME
        self.attribute_channels = attribute_channels
        self.origin_xyz = origin_xyz
        self.vfe1 = VoxelFeatureLayer(attribute_channels + 6, width)
        self.vfe2 = VoxelFeatureLayer(width, width)
        self.context = nn.Sequential(
            ME.MinkowskiConvolution(width, width, kernel_size=3, dimension=3),
            ME.MinkowskiBatchNorm(width), ME.MinkowskiReLU(),
            ME.MinkowskiConvolution(width, width, kernel_size=3, dilation=2, dimension=3),
            ME.MinkowskiBatchNorm(width), ME.MinkowskiReLU(),
            ME.MinkowskiConvolution(width, width, kernel_size=3, dilation=4, dimension=3),
            ME.MinkowskiBatchNorm(width), ME.MinkowskiReLU())

    def forward(self, xyz: torch.Tensor, attrs: torch.Tensor,
                anchor_index: torch.Tensor) -> torch.Tensor:
        import MinkowskiEngine as ME
        batch, count, _ = xyz.shape
        origin = xyz.new_tensor(self.origin_xyz)
        spacing = xyz.new_tensor((0.2, 0.2, 0.25))
        cells = torch.floor((xyz - origin) / spacing).int()
        rows = torch.arange(batch, device=xyz.device, dtype=torch.int32)
        coordinates = torch.cat((rows[:, None, None].expand(batch, count, 1),
                                 cells), -1).reshape(-1, 4)
        unique, inverse = torch.unique(coordinates, dim=0, return_inverse=True)
        flat_xyz = xyz.reshape(-1, 3)
        centers = origin + (cells.float().reshape(-1, 3) + 0.5) * spacing
        features = torch.cat((flat_xyz, attrs.reshape(-1, self.attribute_channels),
                              flat_xyz - centers), -1)
        point, _ = self.vfe1(features, inverse, len(unique))
        _, voxel_features = self.vfe2(point, inverse, len(unique))
        sparse = self.context(ME.SparseTensor(features=voxel_features,
                                              coordinates=unique.int()))
        # Coordinates are queried explicitly: never assume sparse kernels keep
        # the input ordering after their coordinate-map operations.
        contextual = sparse.features_at_coordinates(unique.float())
        point_features = contextual[inverse].reshape(batch, count, -1)
        return gather(point_features, anchor_index)


class AnchorDecoder(nn.Module):
    def __init__(self, width: int, children: int, xyz_half_range: torch.Tensor,
                 columns: int = 5, dropout_p: float = 0.0):
        super().__init__()
        self.num_children = children
        self.columns = columns
        self.register_buffer("xyz_half_range", xyz_half_range.float())
        hidden = [nn.Conv1d(width * 2 + 3, width * 2, 1), nn.ReLU()]
        if dropout_p:
            hidden.append(nn.Dropout1d(dropout_p))
        hidden.extend((nn.Conv1d(width * 2, width * 2, 1), nn.ReLU()))
        if dropout_p:
            hidden.append(nn.Dropout1d(dropout_p))
        hidden.append(nn.Conv1d(width * 2, children * columns, 1))
        self.network = nn.Sequential(*hidden)

    def forward(self, centers: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
        global_feature = features.amax(1, keepdim=True).expand_as(features)
        latent = torch.cat((centers, features, global_feature), -1).transpose(1, 2)
        output = self.network(latent).transpose(1, 2)
        output = output.reshape(centers.shape[0], centers.shape[1], self.num_children,
                                self.columns)
        # Maximum child offset is 2 m in each physical axis.
        delta = torch.tanh(output[..., :3]) * (2 / self.xyz_half_range)
        xyz = (centers[:, :, None] + delta).clamp(-1, 1)
        attrs = torch.tanh(output[..., 3:])
        return torch.cat((xyz, attrs), -1).flatten(1, 2)


class RadarAutoencoder(nn.Module):
    def __init__(self, variant: str, minimum: torch.Tensor, maximum: torch.Tensor,
                 *, width: int = 64, anchors: int = 256, children: int = 4,
                 columns: int = 5, dropout_p: float = 0.0):
        super().__init__()
        if variant not in ("voxelnet", "pointnetpp"):
            raise ValueError(variant)
        if columns < 5:
            raise ValueError("Radar autoencoders need XYZ and at least two attributes")
        if not 0.0 <= dropout_p < 1.0:
            raise ValueError("Dropout probability must be in [0,1)")
        self.variant, self.anchors = variant, anchors
        self.columns = columns
        self.dropout_p = dropout_p
        self.register_buffer("minimum", minimum.float())
        self.register_buffer("maximum", maximum.float())
        self.encoder = (VoxelNetEncoder(
            width, columns - 3,
            tuple(float(value) for value in minimum[:3]) if columns == 7
            else (0.0, -25.6, -3.0)) if variant == "voxelnet"
                        else PointNetPPEncoder(width, columns - 3, dropout_p))
        self.decoder = AnchorDecoder(width, children,
                                     (self.maximum[:3] - self.minimum[:3]) / 2,
                                     columns, dropout_p)

    def forward(self, normalized: torch.Tensor, *, zero_features: bool = False) -> torch.Tensor:
        if normalized.ndim != 3 or normalized.shape[-1] != self.columns:
            raise ValueError(f"Expected [B,N,{self.columns}] radar")
        xyz = (normalized[..., :3] + 1) * (
            (self.maximum[:3] - self.minimum[:3]) / 2) + self.minimum[:3]
        anchors = farthest_points(xyz.detach(), self.anchors)
        features = self.encoder(xyz, normalized[..., 3:], anchors)
        if zero_features:
            features = torch.zeros_like(features)
        return self.decoder(gather(normalized[..., :3], anchors), features)


def reconstruction_loss(predicted: torch.Tensor, target: torch.Tensor,
                        minimum: torch.Tensor, maximum: torch.Tensor) -> dict:
    half_range = (maximum[:3] - minimum[:3]) / 2
    pred_xyz = (predicted[..., :3] + 1) * half_range + minimum[:3]
    true_xyz = (target[..., :3] + 1) * half_range + minimum[:3]
    distances = torch.cdist(pred_xyz, true_xyz)
    pred_dist, nearest = distances.min(-1)
    true_dist = distances.min(-2).values
    chamfer = pred_dist.square().mean() + true_dist.square().mean()
    close = (F.smooth_l1_loss(pred_dist, torch.zeros_like(pred_dist), beta=0.2) +
             F.smooth_l1_loss(true_dist, torch.zeros_like(true_dist), beta=0.2))
    target_attrs = gather(target[..., 3:], nearest)
    attributes = F.mse_loss(predicted[..., 3:], target_attrs)
    total = chamfer + close + 0.05 * attributes
    return {"total": total, "chamfer": chamfer, "close": close,
            "feature": attributes}

"""Physical geometry and separately ablatable Stage-1 model settings."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math


@dataclass(frozen=True)
class VoxelGrid:
    minimum_xyz: tuple[float, float, float] = (-128.0, -128.0, -32.0)
    maximum_xyz: tuple[float, float, float] = (128.0, 128.0, 32.0)
    size_xyz: tuple[float, float, float] = (0.2, 0.2, 0.25)

    def __post_init__(self) -> None:
        for low, high, step in zip(self.minimum_xyz, self.maximum_xyz, self.size_xyz):
            cells = (high - low) / step
            if not all(math.isfinite(v) for v in (low, high, step)) or step <= 0 or cells <= 0 or abs(cells-round(cells)) > 1e-5:
                raise ValueError("Grid limits must contain a whole, positive number of finite voxels")

    @property
    def shape_zyx(self) -> tuple[int, int, int]:
        x, y, z = (round((high-low)/step) for low, high, step in zip(self.minimum_xyz, self.maximum_xyz, self.size_xyz))
        return z, y, x

    def scale_shape(self, stride: int) -> tuple[int, int, int]:
        if stride < 1 or any(n % stride for n in self.shape_zyx):
            raise ValueError("Scale stride must divide all grid dimensions")
        return tuple(n // stride for n in self.shape_zyx)

    def centers_xyz(self, coords_bzyx, stride: int = 1):
        """Return physical centers of sparse cells, preserving input device."""
        import torch
        xyz = coords_bzyx[:, [3, 2, 1]].to(torch.float32)
        origin = xyz.new_tensor(self.minimum_xyz)
        size = xyz.new_tensor(self.size_xyz) * stride
        return origin + (xyz + 0.5) * size


@dataclass(frozen=True)
class Stage1Config:
    grid: VoxelGrid = VoxelGrid()
    channels: tuple[int, ...] = (32, 64, 128, 256)
    encoder_backend: str = "native"  # native keeps existing checkpoints; pfe_spconv uses encoder.py's backbone
    attention_radii_m: tuple[float, ...] = (0.75, 1.5, 3.0, 6.0)
    max_neighbors: int = 24
    attention_dim: int = 64
    growth_scales: tuple[int, ...] = (4,)
    # The defaults are approximately one voxel diagonal at each stride.
    # They define geometric proxies, not verified shared reflector identities.
    positive_radii_m: tuple[float, ...] = (0.4, 0.8, 1.6, 3.2)
    corr_scale_weights: tuple[float, ...] = (0.25, 0.25, 0.25, 0.25)
    bottleneck_cross_attention_only: bool = False
    bottleneck_support_only: bool = False
    bottleneck_dropout: float = 0.0
    # Historical checkpoints generated 64 sites per stride-8 radar cell.
    # New runs score only sites containing observed radar returns.
    support_seed_mode: str = "radar_s2"
    support_stride: int = 2
    support_eval_threshold: float = 0.25
    support_loss_weight: float = 1.0
    support_abstention_weight: float = 0.05
    bottleneck_distill_weight: float = 0.5
    temperature: float = 0.1
    negative_strategy: str = "nearest"  # nearest (hard local) or random_local
    num_negatives: int = 16
    correspondence_weight: float = 1.0
    geometric_weight: float = 0.2
    confidence_weight: float = 0.1
    occupancy_weight: float = 1.0
    local_geometry_weight: float = 1.0
    confidence_sigma_m: float = 0.5
    geometry_eval_tolerance_m: float = 0.2
    # Zero preserves the original checkpoint architecture. The surface config
    # enables multiple radar-conditioned LiDAR locations per fine radar site.
    surface_proposals_per_site: int = 0
    surface_radius_m: float = 1.5
    surface_target_neighbors: int = 24
    surface_geometry_weight: float = 0.0
    surface_confidence_weight: float = 0.0
    surface_match_sigma_m: float = 0.2
    surface_context_radii_m: tuple[float, ...] = (1.0, 2.0, 4.0, 6.0)
    surface_context_neighbors: int = 16
    # Region mode predicts a continuous ellipsoidal support patch per surface
    # proposal. Older surface checkpoints retain their point-only head.
    surface_region_enabled: bool = False
    surface_region_anchor_stride: int = 1
    surface_region_min_radius_xyz_m: tuple[float, float, float] = (0.50, 0.50, 0.625)
    surface_region_max_radius_xyz_m: tuple[float, float, float] = (1.5, 1.5, 1.25)
    surface_region_coverage_weight: float = 0.0
    surface_region_volume_weight: float = 0.0

    def __post_init__(self) -> None:
        if self.encoder_backend not in ("native", "pfe_spconv"):
            raise ValueError("encoder_backend must be native or pfe_spconv")
        if not 1 <= len(self.channels) <= 4 or min(self.channels) < 4:
            raise ValueError("Use 1–4 scales with at least four channels")
        if self.encoder_backend == "pfe_spconv" and len(self.channels) != 4:
            raise ValueError("The PFE sparse encoder requires four feature scales")
        if self.encoder_backend == "pfe_spconv" and self.growth_scales:
            raise ValueError("The PFE sparse encoder has no support-growth convolution; set growth_scales to []")
        if any(b <= a for a,b in zip(self.channels[:-1],self.channels[1:])):
            raise ValueError("Channel widths must increase across scales")
        if len(self.attention_radii_m) < len(self.channels) or any(x <= 0 for x in self.attention_radii_m[:len(self.channels)]):
            raise ValueError("A positive physical radius is required for each scale")
        if len(self.positive_radii_m) < len(self.channels) or any(x <= 0 or x >= self.attention_radii_m[i] for i,x in enumerate(self.positive_radii_m[:len(self.channels)])):
            raise ValueError("Each positive radius must be positive and smaller than its local search radius")
        if len(self.corr_scale_weights) < len(self.channels) or any(x < 0 for x in self.corr_scale_weights[:len(self.channels)]):
            raise ValueError("Each correspondence scale weight must be nonnegative")
        if self.bottleneck_cross_attention_only and (
                len(self.channels) != 4 or any(self.corr_scale_weights[:-1])
                or self.corr_scale_weights[-1] <= 0):
            raise ValueError("Bottleneck correspondence requires four scales and weights [0, 0, 0, positive]")
        if self.bottleneck_support_only:
            if self.support_seed_mode not in ("dense_bottleneck", "radar_s2"):
                raise ValueError("Bottleneck support seed mode must be dense_bottleneck or radar_s2")
            if (self.encoder_backend != "pfe_spconv" or not self.bottleneck_cross_attention_only or len(self.channels) != 4
                    or self.support_stride != 2 or self.surface_proposals_per_site
                    or self.surface_region_enabled or self.confidence_weight):
                raise ValueError("Bottleneck support needs S4 correspondence, stride-2 support, and no legacy proposal/confidence heads")
            if not 0 < self.support_eval_threshold < 1:
                raise ValueError("Support evaluation threshold must be in (0,1)")
            if min(self.support_loss_weight, self.bottleneck_distill_weight) <= 0 or self.support_abstention_weight < 0:
                raise ValueError("Bottleneck support loss weights are invalid")
        if not math.isfinite(self.bottleneck_dropout) or not 0 <= self.bottleneck_dropout < 1:
            raise ValueError("Bottleneck dropout must be in [0, 1)")
        if not math.isfinite(self.temperature) or self.temperature <= 0:
            raise ValueError("Contrastive temperature must be positive")
        if self.negative_strategy not in ("nearest", "random_local") or self.num_negatives < 1:
            raise ValueError("Use nearest or random_local with at least one negative")
        if self.max_neighbors < 1 or self.attention_dim < 4:
            raise ValueError("Attention capacity must be positive")
        if len(set(self.growth_scales)) != len(self.growth_scales) or any(x < 1 or x > len(self.channels) for x in self.growth_scales):
            raise ValueError("Growth scales must be distinct valid one-based scale indices")
        if any(not math.isfinite(x) or x < 0 for x in (self.correspondence_weight, self.geometric_weight, self.confidence_weight, self.occupancy_weight, self.local_geometry_weight)):
            raise ValueError("Loss weights must be nonnegative")
        if self.correspondence_weight+self.geometric_weight+self.confidence_weight+self.surface_geometry_weight+self.surface_confidence_weight == 0:
            raise ValueError("At least one implemented training objective must be enabled")
        if self.confidence_sigma_m <= 0 or self.geometry_eval_tolerance_m <= 0:
            raise ValueError("Confidence sigma and geometry evaluation tolerance must be positive meters")
        if (self.surface_proposals_per_site < 0 or self.surface_target_neighbors < 1
                or self.surface_context_neighbors < 1
                or self.surface_radius_m <= 0 or self.surface_match_sigma_m <= 0
                or min(self.surface_geometry_weight,self.surface_confidence_weight) < 0):
            raise ValueError("Invalid surface proposal settings")
        if len(self.surface_context_radii_m) < len(self.channels) or any(
                radius <= 0 for radius in self.surface_context_radii_m[:len(self.channels)]):
            raise ValueError("Surface context needs a positive physical radius per scale")
        if self.surface_proposals_per_site == 0 and (self.surface_geometry_weight or self.surface_confidence_weight):
            raise ValueError("Surface losses require surface proposals")
        if self.surface_proposals_per_site and self.surface_geometry_weight == 0:
            raise ValueError("Surface proposals need a geometric training objective")
        if self.surface_region_enabled:
            if self.surface_region_anchor_stride < 1 or (self.surface_region_anchor_stride & (self.surface_region_anchor_stride-1)):
                raise ValueError("Region anchor stride must be a positive power of two")
            self.grid.scale_shape(self.surface_region_anchor_stride)
            if self.surface_proposals_per_site == 0 or self.surface_region_coverage_weight <= 0:
                raise ValueError("Learned support regions require proposals and coverage supervision")
            if any(not 0 < low < high for low,high in zip(
                    self.surface_region_min_radius_xyz_m,self.surface_region_max_radius_xyz_m)):
                raise ValueError("Region radii must have positive ordered bounds")
            if any(low < 2*step for low,step in zip(
                    self.surface_region_min_radius_xyz_m,self.grid.size_xyz)):
                raise ValueError("Minimum region radius must span multiple fine voxels")
        elif self.surface_region_coverage_weight or self.surface_region_volume_weight:
            raise ValueError("Region losses require region mode")
        if self.surface_region_volume_weight < 0:
            raise ValueError("Region volume weight must be nonnegative")
        self.grid.scale_shape(2 ** (len(self.channels)-1))

    def as_dict(self) -> dict:
        return asdict(self)

    @property
    def conditioning_channels(self) -> tuple[int, ...]:
        return (self.attention_dim,) if self.bottleneck_support_only else self.channels

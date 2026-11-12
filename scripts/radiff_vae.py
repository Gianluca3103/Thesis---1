"""Paper-guided 4D-RaDiff radar VAE built from pinned D-PCC point blocks.

The publication specifies three density-aware down/up blocks, a 4-D Gaussian
latent point feature, a PointNet-like latent-coordinate decoder, and its loss
weights. Exact unpublished layer widths and fixed training count are explicit
experiment choices, not claimed to be the authors' private implementation.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from types import SimpleNamespace

import torch
from torch import nn


@dataclass(frozen=True)
class RaDiffVAEConfig:
    in_fdim: int = 5
    radar_attribute_dim: int = 2
    latent_dim: int = 4
    compress_normal: bool = True  # D-PCC name for the two radar feature heads.
    k: int = 16
    downsample_rate: tuple[float, ...] = (1 / 3, 1 / 3, 1 / 3)
    max_upsample_num: tuple[int, ...] = (8, 8, 8)
    layer_num: int = 3
    dim: int = 8
    hidden_dim: int = 64
    ngroups: int = 1
    quantize_latent_xyzs: bool = False
    latent_xyzs_conv_mode: str = "mlp"
    sub_point_conv_mode: str = "mlp"
    chamfer_coe: float = 0.1       # Intermediate Chamfer, paper Table 7.
    pts_num_coe: float = 5e-7     # Cardinality, paper Table 7.
    normal_coe: float = 0.05      # Radar feature MSE, paper Table 7.
    mean_distance_coe: float = 50.0
    density_coe: float = 1e-4
    kl_coe: float = 1e-5

    def upstream_args(self) -> SimpleNamespace:
        return SimpleNamespace(**asdict(self))


class RaDiffRadarVAE(nn.Module):
    """Encode XYZ/Doppler/RCS into latent point Gaussians and decode radar."""

    def __init__(self, config: RaDiffVAEConfig = RaDiffVAEConfig()):
        super().__init__()
        # Imported after the professor launcher selects the pinned upstream
        # tree and loads its compiled pointops/Chamfer extensions.
        from models.encoder import Encoder
        from models.decoder import Decoder

        self.config = config
        self.args = config.upstream_args()
        self.pre_conv = nn.Sequential(
            nn.Conv1d(5, config.hidden_dim, 1),
            nn.GroupNorm(config.ngroups, config.hidden_dim), nn.ReLU(),
            nn.Conv1d(config.hidden_dim, config.dim, 1))
        self.encoder = Encoder(self.args)
        self.mu = nn.Sequential(nn.Conv1d(config.dim + 3, config.hidden_dim, 1),
                                nn.ReLU(), nn.Conv1d(config.hidden_dim,
                                                     config.latent_dim, 1))
        self.logvar = nn.Sequential(nn.Conv1d(config.dim + 3, config.hidden_dim, 1),
                                    nn.ReLU(), nn.Conv1d(config.hidden_dim,
                                                         config.latent_dim, 1))
        # Figure 5 has a PointNet branch for latent XYZ and an MLP feature
        # branch. The pointwise-plus-global PointNet widths are unpublished.
        self.xyz_pointnet = nn.Sequential(
            nn.Conv1d(config.latent_dim, config.hidden_dim, 1), nn.ReLU(),
            nn.Conv1d(config.hidden_dim, config.hidden_dim, 1), nn.ReLU())
        self.xyz_out = nn.Sequential(
            nn.Conv1d(config.hidden_dim * 2, config.hidden_dim, 1), nn.ReLU(),
            nn.Conv1d(config.hidden_dim, 3, 1), nn.Tanh())
        self.feat_out = nn.Sequential(
            nn.Conv1d(config.latent_dim, config.hidden_dim, 1), nn.ReLU(),
            nn.Conv1d(config.hidden_dim, config.dim, 1))
        self.decoder = Decoder(self.args)

    def _reset_neighbors(self) -> None:
        # Upstream DownsampleLayer mutates k on small frames. Restore the
        # configured k before every independent frame/batch.
        for layer in self.encoder.encoder_layers:
            layer.k = self.config.k

    def forward(self, points: torch.Tensor, *, sample_latent: bool | None = None) -> dict:
        from models.loss import get_chamfer_loss, get_pts_num_loss
        from models.utils import index_points

        if points.ndim != 3 or points.shape[1] != 5 or points.shape[2] < 14:
            raise ValueError("Expected normalized radar [B,5,N] with at least 14 points")
        self._reset_neighbors()
        xyz, radar_features = points[:, :3], points[:, 3:]
        encoded = self.pre_conv(points)
        gt_xyzs, gt_dnums, gt_mdis, latent_xyz, latent_feats = self.encoder(
            xyz, encoded)
        moments_input = torch.cat((latent_xyz, latent_feats), dim=1)
        mu = self.mu(moments_input)
        logvar = self.logvar(moments_input).clamp(-12.0, 8.0)
        if sample_latent is None:
            sample_latent = self.training
        z = mu + torch.exp(0.5 * logvar) * torch.randn_like(mu) if sample_latent else mu
        point_features = self.xyz_pointnet(z)
        context = point_features.amax(dim=2, keepdim=True).expand_as(point_features)
        decoded_latent_xyz = self.xyz_out(torch.cat((point_features, context), 1))
        decoded_latent_feats = self.feat_out(z)
        pred_xyzs, pred_unums, pred_mdis, decoded_features = self.decoder(
            decoded_latent_xyz, decoded_latent_feats)
        pred_features = torch.tanh(decoded_features)
        chamfer, pred_to_gt = get_chamfer_loss(gt_xyzs, pred_xyzs, self.args)
        # The paper derives decoded latent XYZ from sampled z, so the first
        # density stage cannot assume the encoded and decoded centers coincide.
        nearest_latent = torch.cdist(decoded_latent_xyz.transpose(1, 2),
                                     latent_xyz.transpose(1, 2)).argmin(-1)
        density = points.new_zeros(())
        for stage in range(self.config.layer_num):
            target_stage = self.config.layer_num - 1 - stage
            nearest = nearest_latent if stage == 0 else pred_to_gt[stage - 1]
            target_counts = index_points(gt_dnums[target_stage].unsqueeze(1),
                                         nearest).squeeze(1)
            target_distance = index_points(gt_mdis[target_stage].unsqueeze(1),
                                           nearest).squeeze(1)
            density = density + (pred_unums[stage] - target_counts).abs().mean()
            density = density + self.config.mean_distance_coe * (
                pred_mdis[stage] - target_distance).abs().mean()
        density = density * self.config.density_coe
        cardinality = get_pts_num_loss(gt_xyzs, pred_unums, self.args)
        target_features = index_points(radar_features, pred_to_gt[-1])
        feature = ((pred_features - target_features).square().sum(dim=1).mean()
                   * self.config.normal_coe)
        kl = (-0.5 * (1 + logvar - mu.square() - logvar.exp())).mean()
        kl = kl * self.config.kl_coe
        loss = chamfer + density + cardinality + feature + kl
        return {"points": torch.cat((pred_xyzs[-1], pred_features), dim=1),
                "latent": z, "mu": mu, "logvar": logvar, "loss": loss,
                "terms": {"chamfer": chamfer, "density": density,
                          "cardinality": cardinality, "feature": feature,
                          "kl": kl}}

"""Radar-only fine-voxel associations for the Stage-I probability map.

Clean LiDAR supplies training targets only. At inference every active 0.2 x
0.2 x 0.25 m radar voxel proposes at most 32 LiDAR voxels within 3 m.
Distinct radar anchors voting for one target increase its probability.
"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import torch
from torch import nn

from .encoder import GRID
from .previous_encoder import PreviousStage1Sparse
from .surface_student import (RadarSurfaceHead, SurfaceStudentConfig,
                              SurfaceStudentOutput)
from .teacher import ResidualAdapter


VERSION = "crossmodal_fine_association_v1"


class FineAssociationStudent(nn.Module):
    def __init__(self, config: SurfaceStudentConfig):
        super().__init__()
        if (not config.pointer_voxel_mode or not config.decoder_from_conditioned
                or not config.spherical_offsets or config.anchor_stride != 1
                or config.proposals_per_anchor not in (8, 32)
                or config.offset_limit_m != 3.0
                or config.target_radius_m != 3.0
                or config.target_neighbors != config.proposals_per_anchor
                or max(config.context_radii_m) > 3.0):
            raise ValueError("Fine association requires 8 or 32 fine radar pointers within 3 m")
        self.config = config
        self.radar_encoder = PreviousStage1Sparse(None, modality="radar")
        self.shared = ResidualAdapter(config.feature_dim)
        self.feature_head = ResidualAdapter(config.feature_dim)
        self.feature_norm = nn.LayerNorm(config.feature_dim)
        self.surface_head = RadarSurfaceHead(config)

    def initialize_from_student(self, saved: dict) -> None:
        if saved.get("version") != "crossmodal_student_v5_multiscale_refined_raw64":
            raise ValueError("Expected the current trained radar-only Stage-I student")
        source = saved["model"]
        for name in ("radar_encoder", "shared", "feature_head", "feature_norm"):
            prefix = name + "."
            state = {key[len(prefix):]: value for key, value in source.items()
                     if key.startswith(prefix)}
            getattr(self, name).load_state_dict(state, strict=True)

    def initialize_from_eight_slot_checkpoint(self, saved: dict,
                                              source_student_sha256: str) -> None:
        """Keep the trained eight slots and initialize the other 24 anew."""
        if (saved.get("version") != VERSION
                or saved.get("source_student_sha256") != source_student_sha256
                or saved.get("config", {}).get("proposals_per_anchor") != 8
                or self.config.proposals_per_anchor != 32):
            raise ValueError("Eight-slot warm start has an incompatible source or config")
        old_config = saved["config"]
        for field in ("anchor_stride", "offset_limit_m", "context_radii_m",
                      "context_neighbors", "feature_dim", "spherical_offsets",
                      "decoder_from_conditioned", "pointer_voxel_mode"):
            if old_config[field] != asdict(self.config)[field]:
                raise ValueError(f"Eight-slot warm start differs in {field}")
        current = self.state_dict()
        prior = saved["model"]
        for key, target in current.items():
            if key.startswith("surface_head.slots."):
                continue
            if key not in prior or target.shape != prior[key].shape:
                raise ValueError(f"Eight-slot checkpoint differs at {key}")
            target.copy_(prior[key])
        width = self.surface_head.slots.in_features
        current["surface_head.slots.weight"].view(32, 4, width)[:8].copy_(
            prior["surface_head.slots.weight"].view(8, 4, width))
        current["surface_head.slots.bias"].view(32, 4)[:8].copy_(
            prior["surface_head.slots.bias"].view(8, 4))
        self.load_state_dict(current, strict=True)

    def forward(self, radar_points: torch.Tensor,
                radar_valid: torch.Tensor) -> SurfaceStudentOutput:
        coarse, levels = self.radar_encoder.forward_pyramid(radar_points, radar_valid)
        shared = coarse.replace_features(self.shared(coarse.features))
        conditioned = shared.replace_features(
            self.feature_norm(self.feature_head(shared.features)))
        proposals = self.surface_head((*levels, conditioned))
        return SurfaceStudentOutput(conditioned, proposals, full_confidence_field=True)


def load_fine_association(path: str | Path, device: str = "cuda"):
    saved = torch.load(path, map_location="cpu", weights_only=False)
    if saved.get("version") != VERSION:
        raise ValueError("Expected a fine radar-to-LiDAR association checkpoint")
    config = dict(saved["config"])
    config["context_radii_m"] = tuple(config["context_radii_m"])
    model = FineAssociationStudent(SurfaceStudentConfig(**config))
    model.load_state_dict(saved["model"], strict=True)
    return model.to(device).eval(), saved


def checkpoint_payload(model: FineAssociationStudent, *, source: dict,
                       source_path: Path, epoch: int, validation: dict) -> dict:
    return {"version": VERSION, "config": asdict(model.config),
            "model": model.state_dict(), "epoch": epoch,
            "source_student_checkpoint": str(source_path.resolve()),
            "radar_variant": source["radar_variant"],
            "lidar_fov": source.get("lidar_fov"),
            "radius_m": 3.0, "validation": validation,
            "grid": {"origin_xyz_m": GRID.minimum_xyz,
                     "voxel_xyz_m": GRID.size_xyz}}

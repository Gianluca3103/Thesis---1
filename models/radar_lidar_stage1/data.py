"""Official synchronized VoD radar and clean-LiDAR pairs, with no fault dependency."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from .config import VoxelGrid
from .online_augmentation import GeometricAugmentationConfig, augment_pair, frame_generator, sample_transform
from .radar_fov import crop_lidar_to_radar_fov

from Fault_Localization_Model.vod_dataset.vod_io import (
    align_radar_to_lidar, discover_vod_frames, load_vod_lidar,
    load_vod_radar, load_vod_radar_to_lidar,
)


class VoDStage1Dataset(Dataset):
    def __init__(self, root: str | Path, split: str, *, radar_variant: str = "radar_20frames_verified_doppler_radial", frame_ids: list[str] | None = None, include_clean: bool = True,
                 augmentation_config: GeometricAugmentationConfig | None = None,
                 augmentation_seed: int = 0, augmentation_grid: VoxelGrid | None = None,
                 lidar_radar_fov_deg: float | None = None,
                 lidar_radar_max_range_m: float | None = None):
        self.frames = discover_vod_frames(root, split, radar_variant=radar_variant, frame_ids=frame_ids)
        self.include_clean = include_clean
        if (lidar_radar_fov_deg is None) != (lidar_radar_max_range_m is None):
            raise ValueError("LiDAR radar-sector angle and range must be set together")
        self.lidar_radar_fov_deg = lidar_radar_fov_deg
        self.lidar_radar_max_range_m = lidar_radar_max_range_m
        self.augmentation_config = augmentation_config or GeometricAugmentationConfig()
        self.augmentation_config.validate()
        if self.augmentation_config.enabled and (split != "train" or not include_clean):
            raise ValueError("Stage-I online augmentation requires paired training radar and clean LiDAR")
        self.augmentation_seed = int(augmentation_seed)
        self.augmentation_grid = augmentation_grid or VoxelGrid()
        self._augmentation_epoch = torch.zeros((), dtype=torch.int64).share_memory_()

    def set_epoch(self, epoch: int) -> None:
        if int(epoch) < 0:
            raise ValueError("augmentation epoch must be nonnegative")
        self._augmentation_epoch.fill_(int(epoch))

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, index: int) -> dict:
        frame = self.frames[index]
        radar = load_vod_radar(frame.radar_path)
        lidar_from_radar = load_vod_radar_to_lidar(
            frame.lidar_calibration_path, frame.radar_calibration_path)
        radar = align_radar_to_lidar(radar, lidar_from_radar)
        if not np.isfinite(radar).all():
            raise ValueError(f"Aligned VoD radar contains NaN or Inf: {frame.radar_path}")
        result = {"radar": torch.from_numpy(radar.copy()), "frame_id": frame.frame_id, "split": frame.split}
        if self.include_clean:
            lidar = load_vod_lidar(frame.lidar_path)
            if self.lidar_radar_fov_deg is not None:
                lidar = crop_lidar_to_radar_fov(
                    lidar, lidar_from_radar,
                    half_fov_deg=self.lidar_radar_fov_deg,
                    max_range_m=self.lidar_radar_max_range_m)
            result["clean_lidar"] = torch.from_numpy(lidar.copy())
        if self.augmentation_config.enabled:
            generator = frame_generator(self.augmentation_seed, int(self._augmentation_epoch),
                                        frame.split, frame.frame_id)
            transform = sample_transform(self.augmentation_config, generator=generator)
            result["radar"], result["clean_lidar"] = augment_pair(
                result["radar"], result["clean_lidar"], transform, self.augmentation_grid)
            result["augmentation_transform"] = transform.to_dict()
        return result


def collate_stage1(batch: list[dict]) -> dict:
    if not batch or len({"clean_lidar" in row for row in batch}) != 1:
        raise ValueError("Batch must be nonempty and uniformly teacher-labeled")
    result = {"frame_id": [row["frame_id"] for row in batch], "split": [row["split"] for row in batch]}
    if any("augmentation_transform" in row for row in batch):
        if not all("augmentation_transform" in row for row in batch):
            raise ValueError("Augmentation metadata must be present for the whole batch")
        result["augmentation_transform"] = [row["augmentation_transform"] for row in batch]
    for name in ("radar", "clean_lidar"):
        if name not in batch[0]:
            continue
        maximum = max(len(row[name]) for row in batch)
        columns = batch[0][name].shape[-1]
        values = torch.zeros(len(batch), maximum, columns, dtype=torch.float32)
        mask = torch.zeros(len(batch), maximum, dtype=torch.bool)
        for i,row in enumerate(batch):
            count = len(row[name])
            values[i,:count] = row[name]
            mask[i,:count] = True
        result[name], result[name+"_valid"] = values, mask
    return result

"""Official VoD pairs and frame-level object-supervision metadata."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

from Fault_Localization_Model.vod_dataset.vod_io import load_vod_lidar_to_camera
from models.radar_lidar_stage1.data import VoDStage1Dataset, collate_stage1
from models.radar_lidar_stage1.online_augmentation import GeometricAugmentationConfig

from .encoder import GRID
from .supervision import voxel_object_ids


class PairedRadarLidarDataset(Protocol):
    """Minimum contract for future non-VoD cross-modal data adapters."""

    frames: list

    def __len__(self) -> int: ...

    def __getitem__(self, index: int) -> dict: ...

    def set_epoch(self, epoch: int) -> None: ...


def tj4dradset_dataset(*args, **kwargs):
    raise NotImplementedError(
        "TJ4DRadSet needs synchronized radar/LiDAR calibration and split loaders, "
        "per-point radar time indices, and oriented 3D instance-box parsing before "
        "this VoD-supervised pipeline can train or evaluate on it.")


def paired_dataset(vod_root: str | Path, split: str, radar_variant: str, *,
                   augmentation: GeometricAugmentationConfig | None = None,
                   seed: int = 0, limit: int | None = None,
                   include_clean: bool = True):
    dataset = VoDStage1Dataset(vod_root, split, radar_variant=radar_variant,
                               include_clean=include_clean,
                               augmentation_config=augmentation,
                               augmentation_grid=GRID, augmentation_seed=seed)
    if limit is not None:
        dataset.frames = dataset.frames[:limit]
    return dataset


def annotate_batch(dataset: VoDStage1Dataset, batch: dict, radar_sites, lidar_sites):
    frames = {frame.frame_id: frame for frame in dataset.frames}
    selected = [frames[frame_id] for frame_id in batch["frame_id"]]
    labels = [frame.lidar_path.parent.parent / "label_2" / f"{frame.frame_id}.txt"
              for frame in selected]
    for path in labels:
        if not path.is_file():
            raise FileNotFoundError(f"VoD object annotations required for teacher metrics: {path}")
    transforms = batch.get("augmentation_transform")
    camera = [load_vod_lidar_to_camera(frame.lidar_calibration_path) for frame in selected]
    radar_ids, radar_classes = voxel_object_ids(
        radar_sites, batch["radar"], batch["radar_valid"], labels, camera,
        transforms, max_radar_age=0.1)
    lidar_ids, lidar_classes = voxel_object_ids(
        lidar_sites, batch["clean_lidar"], batch["clean_lidar_valid"],
        labels, camera, transforms)
    if any(radar_classes.get(identity) != name for identity, name in lidar_classes.items()
           if identity in radar_classes):
        raise ValueError("Radar/LiDAR object-class identity mismatch")
    return radar_ids, lidar_ids, {**lidar_classes, **radar_classes}


__all__ = ["paired_dataset", "collate_stage1", "annotate_batch"]

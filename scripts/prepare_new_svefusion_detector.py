"""Prepare a fresh clean-LiDAR plus five-frame-radar SVEFusion training run.

Uses raw VoD sensor files through symlinks. Only KITTI-normalized labels and
train/validation information pickles are written to the new data root.
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import json
import pickle
from pathlib import Path

import yaml

from Fault_Localization_Model.vod_dataset.vod_io import (
    load_vod_split_ids, resolve_vod_public_root,
)
from scripts.prepare_stage2_svefusion import _link_directory, normalize_vod_labels


def _write_or_check(path: Path, content: str) -> None:
    if path.is_file():
        if path.read_text(encoding="utf-8") != content:
            raise FileExistsError(f"Existing clean detector data differs: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vod-root", type=Path, required=True)
    parser.add_argument("--sve-root", type=Path, required=True)
    parser.add_argument("--stage2-export-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    public = resolve_vod_public_root(args.vod_root).resolve()
    sve = args.sve_root.resolve()
    root = args.data_root.resolve()
    export = json.loads((args.stage2_export_root / "export_manifest.json").read_text(
        encoding="utf-8"))
    if not export["complete"] or export["radar_variant"] != "radar_5frames":
        raise ValueError("Complete five-frame Stage-II validation export required")
    splits = {name: load_vod_split_ids(public, name) for name in ("train", "val")}
    if splits["val"] != export["frame_ids"] or set(splits["train"]) & set(splits["val"]):
        raise ValueError("VoD split differs from the Stage-II comparison export")

    training = root / "training"
    source_training = public / "lidar" / "training"
    sensor_roots = {
        "calib": source_training / "calib",
        "lidar_calib": source_training / "calib",
        "image_2": source_training / "image_2",
        "lidar": source_training / "velodyne",
        "radar_5f": Path(export["radar_points_root"]),
        "radar_calib": Path(export["radar_calibration_root"]),
    }
    for name, source in sensor_roots.items():
        _link_directory(source, training / name)
    labels = training / "label_2"
    labels.mkdir(parents=True, exist_ok=True)
    for split_ids in splits.values():
        for frame_id in split_ids:
            source = source_training / "label_2" / f"{frame_id}.txt"
            _write_or_check(labels / source.name,
                            normalize_vod_labels(source.read_text(encoding="utf-8")))
    for name, ids in splits.items():
        _write_or_check(root / "ImageSets" / f"{name}.txt", "\n".join(ids) + "\n")

    base = sve / "tools" / "cfgs" / "dataset_configs" / "Vod_fusion.yaml"
    model_source = sve / "tools" / "cfgs" / "VoD_models" / "SVEFusion.yaml"
    with base.open(encoding="utf-8") as source:
        dataset = yaml.safe_load(source)
    with model_source.open(encoding="utf-8") as source:
        model = yaml.safe_load(source)
    sensor_config = {
        "DATA_PATH": str(root),
        "LIDAR_POINTS_ROOT": str(training / "lidar"),
        "LIDAR_CALIB_ROOT": str(training / "lidar_calib"),
        "RADAR_POINTS_ROOT": str(training / "radar_5f"),
        "RADAR_CALIB_ROOT": str(training / "radar_calib"),
        "INFO_PATH": {"train": ["vod_infos_train.pkl"],
                      "test": ["vod_infos_val.pkl"]},
        "FOV_POINTS_ONLY": False,
    }
    dataset.update(sensor_config)
    dataset["DATA_AUGMENTOR"]["DISABLE_AUG_LIST"] = ["placeholder", "gt_sampling"]
    model["DATA_CONFIG"].update(copy.deepcopy(sensor_config))
    model["DATA_CONFIG"]["DATA_AUGMENTOR"]["DISABLE_AUG_LIST"] = [
        "placeholder", "gt_sampling"]
    model_path = sve / "tools" / "cfgs" / "VoD_models" / "SVEFusion_stage2_new_clean.yaml"
    model_text = yaml.safe_dump(model, sort_keys=False)
    _write_or_check(model_path, model_text)

    from easydict import EasyDict
    from pcdet.datasets.vod.vod_dataset import VodDataset

    data = VodDataset(EasyDict(dataset), ["Car", "Pedestrian", "Cyclist"],
                      training=False, root_path=root)
    log_path = root / "preparation.log"
    for split in ("train", "val"):
        path = root / f"vod_infos_{split}.pkl"
        if path.is_file():
            with path.open("rb") as source:
                infos = pickle.load(source)
        else:
            data.set_split(split)
            with log_path.open("a", encoding="utf-8") as log, contextlib.redirect_stdout(log):
                infos = data.get_infos(num_workers=args.workers, has_label=True,
                                       count_inside_pts=False)
            temporary = path.with_suffix(".pkl.tmp")
            with temporary.open("wb") as target:
                pickle.dump(infos, target, protocol=pickle.HIGHEST_PROTOCOL)
            temporary.replace(path)
        actual = [str(item["point_cloud"]["lidar_idx"]) for item in infos]
        if actual != splits[split]:
            raise ValueError(f"{split} detector info IDs differ from official VoD")
        print(f"{split}: {len(infos)} clean LiDAR/radar detector frames", flush=True)
    _write_or_check(root / "preparation_manifest.json", json.dumps({
        "version": 1, "vod_root": str(public),
        "splits": {name: len(ids) for name, ids in splits.items()},
        "model_config": str(model_path),
        "stage2_export_root": str(args.stage2_export_root.resolve()),
    }, indent=2) + "\n")
    print(f"Clean SVEFusion training config: {model_path}")


if __name__ == "__main__":
    main()

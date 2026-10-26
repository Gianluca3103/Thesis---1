"""Point a trained SVEFusion config at matched Stage-II LiDAR exports.

Run this with the SVEFusion Python environment. The detector model and all
dataset preprocessing settings are copied from the supplied training config.
"""

from __future__ import annotations

import argparse
import copy
import filecmp
import json
import pickle
from pathlib import Path

import numpy as np
import yaml

from Fault_Localization_Model.vod_dataset.vod_io import (
    load_vod_lidar_to_camera, load_vod_radar_to_lidar,
)


CONDITIONS = ("clean", "faulty", "reconstructed")


def _validation_infos(data_root: Path, dataset: dict) -> list[dict]:
    infos = []
    for name in dataset["INFO_PATH"]["test"]:
        path = data_root / name
        if path.is_file():
            with path.open("rb") as source:
                infos.extend(pickle.load(source))
    if not infos:
        raise FileNotFoundError(f"No validation info pickle under {data_root}")
    return infos


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sve-root", type=Path, required=True)
    parser.add_argument("--detector-config", type=Path, required=True,
                        help="Exact SVEFusion model YAML used to train the checkpoint")
    parser.add_argument("--detector-data-root", type=Path,
                        help="Override DATA_PATH only if the training config used a placeholder")
    parser.add_argument("--export-root", type=Path, required=True)
    args = parser.parse_args()
    sve = args.sve_root.resolve()
    with args.detector_config.open(encoding="utf-8") as source:
        model = yaml.safe_load(source)
    reference = Path(model["DATA_CONFIG"]["_BASE_CONFIG_"])
    dataset_path = reference if reference.is_absolute() else sve / "tools" / reference
    with dataset_path.open(encoding="utf-8") as source:
        dataset = yaml.safe_load(source)
    for key, value in model["DATA_CONFIG"].items():
        if key not in ("_BASE_CONFIG_", "LIDAR_POINTS_ROOT"):
            dataset[key] = value
    if dataset.get("SENSOR") != "Fusion" or dataset["DATA_SPLIT"]["test"] != "val":
        raise ValueError("Detector config must be LiDAR+radar fusion on validation")
    data_root = (args.detector_data_root or Path(dataset["DATA_PATH"])).resolve()
    if not data_root.is_dir():
        raise FileNotFoundError(f"Detector data root does not exist: {data_root}")
    infos = _validation_infos(data_root, dataset)
    actual_ids = [str(item["point_cloud"]["lidar_idx"]) for item in infos]
    manifest = json.loads((args.export_root / "export_manifest.json").read_text(encoding="utf-8"))
    ids = manifest["frame_ids"]
    if not manifest["complete"]:
        raise ValueError("Stage-II validation export is incomplete")
    if len(actual_ids) != len(set(actual_ids)) or not set(actual_ids).issubset(ids):
        vod_info_path = data_root / "vod_infos_val.pkl"
        if not vod_info_path.is_file():
            raise ValueError("Detector validation IDs/order differ from official VoD")
        with vod_info_path.open("rb") as source:
            infos = pickle.load(source)
        actual_ids = [str(item["point_cloud"]["lidar_idx"]) for item in infos]
        if len(actual_ids) != len(set(actual_ids)) or not set(actual_ids).issubset(ids):
            raise ValueError("Detector VoD validation IDs/order differ from Stage II")
        # Some training configs list a second dataset's infos as well. Evaluate
        # the official VoD validation split only, with unchanged preprocessing.
        model["DATA_CONFIG"].setdefault("INFO_PATH", {})["test"] = ["vod_infos_val.pkl"]
    raw_root = Path(manifest["clean_points_root"])
    stage1_radar_root = Path(manifest["radar_points_root"])
    stage1_lidar_calib = Path(manifest["lidar_calibration_root"])
    stage1_radar_calib = Path(manifest["radar_calibration_root"])
    detector_clean_root = Path(dataset.get(
        "LIDAR_POINTS_ROOT", data_root / "training" / "lidar"))
    radar_root = Path(dataset.get("RADAR_POINTS_ROOT", data_root / "training" / "radar_5f"))
    lidar_calib = Path(dataset.get("LIDAR_CALIB_ROOT", data_root / "training" / "lidar_calib"))
    radar_calib = Path(dataset.get("RADAR_CALIB_ROOT", data_root / "training" / "radar_calib"))
    detector_label_calib = data_root / "training" / "calib"
    point_range = dataset["POINT_CLOUD_RANGE"]
    if point_range[0] < 0:
        raise ValueError("This comparison expects SVEFusion's forward-only detector range")
    clean_policies = set()
    for frame_id in actual_ids:
        for path in (raw_root / f"{frame_id}.bin",
                     detector_clean_root / f"{frame_id}.bin",
                     args.export_root / "faulty" / f"{frame_id}.bin",
                     args.export_root / "reconstructed" / f"{frame_id}.bin",
                     radar_root / f"{frame_id}.bin",
                     lidar_calib / f"{frame_id}.txt",
                     radar_calib / f"{frame_id}.txt",
                     detector_label_calib / f"{frame_id}.txt"):
            if not path.is_file():
                raise FileNotFoundError(path)
        raw_path = raw_root / f"{frame_id}.bin"
        detector_path = detector_clean_root / f"{frame_id}.bin"
        if not filecmp.cmp(raw_path, detector_path, shallow=False):
            raw = np.fromfile(raw_path, dtype="<f4").reshape(-1, 4)
            trained = np.fromfile(detector_path, dtype="<f4").reshape(-1, 4)
            forward = raw[raw[:, 0] >= 0]
            sentinel = np.array([[0.01, 0.0, -2.9, 0.0]], dtype=np.float32)
            if not np.array_equal(trained, forward if len(forward) else sentinel):
                raise ValueError(f"Detector clean scan differs from raw/forward VoD: {frame_id}")
            clean_policies.add("forward_only")
        else:
            raw = np.fromfile(raw_path, dtype="<f4").reshape(-1, 4)
            if np.any(raw[:, 0] < 0):
                clean_policies.add("full_scan")
        for source, target, suffix, label in (
            (stage1_radar_root, radar_root, "bin", "radar"),
        ):
            if not filecmp.cmp(source / f"{frame_id}.{suffix}",
                               target / f"{frame_id}.{suffix}", shallow=False):
                raise ValueError(f"Detector {label} differs from Stage-II VoD: {frame_id}")
        source_lidar = stage1_lidar_calib / f"{frame_id}.txt"
        source_radar = stage1_radar_calib / f"{frame_id}.txt"
        target_lidar = lidar_calib / f"{frame_id}.txt"
        target_radar = radar_calib / f"{frame_id}.txt"
        if (not np.allclose(load_vod_lidar_to_camera(source_lidar),
                            load_vod_lidar_to_camera(detector_label_calib / f"{frame_id}.txt"),
                            rtol=0, atol=1e-6)
                or not np.allclose(load_vod_radar_to_lidar(source_lidar, source_radar),
                                   load_vod_radar_to_lidar(target_lidar, target_radar),
                                   rtol=0, atol=1e-6)):
            raise ValueError(f"Detector calibration differs from Stage-II VoD: {frame_id}")
    if len(clean_policies) > 1:
        raise ValueError("Detector clean export mixes full and forward-only LiDAR policies")
    paths = {
        "clean": detector_clean_root,
        "faulty": (args.export_root / "faulty").resolve(),
        "reconstructed": (args.export_root / "reconstructed").resolve(),
    }
    target_dir = sve / "tools" / "cfgs" / "VoD_models"
    if not target_dir.is_dir():
        raise FileNotFoundError(target_dir)
    with (args.export_root / "svefusion_val_infos.pkl").open("wb") as target:
        pickle.dump(infos, target, protocol=pickle.HIGHEST_PROTOCOL)
    for condition in CONDITIONS:
        config = copy.deepcopy(model)
        config["DATA_CONFIG"]["LIDAR_POINTS_ROOT"] = str(paths[condition])
        config["DATA_CONFIG"]["DATA_PATH"] = str(data_root)
        path = target_dir / f"SVEFusion_stage2_{condition}.yaml"
        path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
        print(f"{condition}: {path}")
    print(f"Validated {len(actual_ids)} common VoD frames, one fixed radar source, "
          f"and clean policy {next(iter(clean_policies), 'ambiguous')}")


if __name__ == "__main__":
    main()

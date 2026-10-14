"""Verify every official train/val pair before cross-modal training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from Fault_Localization_Model.vod_dataset.vod_io import resolve_vod_public_root
from models.radar_lidar_crossmodal_v1.pipeline_data import paired_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vod-root", required=True, type=Path)
    parser.add_argument("--radar-variant", required=True)
    args = parser.parse_args()
    public = resolve_vod_public_root(args.vod_root)

    for split in ("train", "val"):
        manifests = (
            (
                public / "radar_20frames_verified" / f"filter_manifest_{split}.json",
                {
                    "variant": "radar_20frames_verified",
                    "stack_size": 20,
                    "alignment_source": "official-5",
                    "basic_validity_filter": False,
                    "temporal_filter": False,
                    "isolate_split_history": True,
                },
            ),
            (
                public / args.radar_variant / f"doppler_manifest_{split}.json",
                {
                    "source_variant": "radar_20frames_verified",
                    "output_variant": args.radar_variant,
                    "frame_period_s": 0.10022,
                    "doppler_sign": 1,
                    "dispersion_tolerance_m": None,
                    "ego_alignment": "verified_official_5",
                },
            ),
        )
        for path, expected in manifests:
            if not path.is_file():
                raise FileNotFoundError(f"Missing cache provenance manifest: {path}")
            saved = json.loads(path.read_text(encoding="utf-8"))
            mismatched = {
                key: (saved.get(key), value)
                for key, value in expected.items()
                if saved.get(key) != value
            }
            if mismatched:
                raise ValueError(f"Cache provenance mismatch in {path}: {mismatched}")

        dataset = paired_dataset(args.vod_root, split, args.radar_variant)
        malformed_radar = [
            frame.frame_id
            for frame in dataset.frames
            if frame.radar_path.stat().st_size == 0
            or frame.radar_path.stat().st_size % 28 != 0
        ]
        if malformed_radar:
            raise ValueError(
                f"VoD {split} has {len(malformed_radar)} empty or malformed Doppler "
                f"radar frames; first: {malformed_radar[0]}"
            )
        missing_labels = [
            frame.frame_id
            for frame in dataset.frames
            if not (
                frame.lidar_path.parent.parent / "label_2" / f"{frame.frame_id}.txt"
            ).is_file()
        ]
        if missing_labels:
            raise FileNotFoundError(
                f"VoD {split} has {len(missing_labels)} frames without object labels; "
                f"first: {missing_labels[0]}"
            )
        print(f"Verified {split}: {len(dataset)} complete radar/LiDAR/label pairs", flush=True)


if __name__ == "__main__":
    main()

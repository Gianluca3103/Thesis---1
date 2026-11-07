"""Rebuild complete voxel viewers for frames selected by a geometry report.

This reads the existing metric report and recomputes only its viewer frames.
It does not rerun the full validation comparison or any model checkpoint.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from models.radar_lidar_crossmodal_v1.hierarchical_gate import missing_target, quantize
from models.radar_lidar_crossmodal_v1.pipeline_data import paired_dataset
from models.radar_lidar_stage2.fault_data import load_faulty
from models.radar_lidar_stage2.mild_fault_data import MILD_FOG2_VERSION
from scripts.compare_radar_support_rules import (
    RULE_NAMES, _viewer_index, flat_indices, nearest_distances, write_viewer_frame,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--vod-root", type=Path, required=True)
    parser.add_argument("--data-cache-root", type=Path, required=True)
    parser.add_argument("--faulty-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--radar-variant", default="radar_20frames_verified_doppler_radial")
    parser.add_argument("--display-points", type=int, default=5000)
    args = parser.parse_args()
    if args.display_points < 1:
        parser.error("--display-points must be positive")
    report = json.loads(args.report.read_text(encoding="utf-8"))
    report_names = [row["name"] for row in report["rules"]]
    if not report_names or not set(report_names) <= set(RULE_NAMES):
        parser.error("Report uses unknown geometry rules")
    viewers = report.get("viewer_frames")
    if not viewers:
        parser.error("Report has no selected viewer frames")
    dataset = paired_dataset(args.vod_root, "val", args.radar_variant,
                             include_clean=True, data_cache_root=args.data_cache_root)
    manifest = json.loads((args.faulty_root / "val" / "manifest.json").read_text(
        encoding="utf-8"))
    if (manifest.get("version") != MILD_FOG2_VERSION or
            manifest.get("radar_variant") != args.radar_variant or
            manifest.get("frame_ids") != [frame.frame_id for frame in dataset.frames]):
        parser.error("Mild-fog2 cache differs from validation dataset")
    frame_index = {frame.frame_id: i for i, frame in enumerate(dataset.frames)}
    args.output_root.mkdir(parents=True, exist_ok=True)
    output_rows = []
    for viewer in viewers:
        frame_id = viewer["frame_id"]
        fault = viewer["fault"]
        if frame_id not in frame_index or manifest["frame_faults"][frame_id] != fault:
            parser.error(f"Viewer frame/fault not in validation set: {frame_id}")
        row = dataset[frame_index[frame_id]]
        faulty = load_faulty(args.faulty_root, "val", frame_id)
        observed = quantize(faulty, 1)
        missing = missing_target(row["clean_lidar"][:, :3], faulty)
        distances = nearest_distances(row["radar"])
        filename = f"{frame_id}_{fault}.html"
        write_viewer_frame(args.output_root / filename, frame_id=frame_id,
                           fault=fault, distances=distances,
                           observed_index=flat_indices(observed),
                           clean=quantize(row["clean_lidar"][:, :3], 1),
                           faulty=observed, missing=missing, radar=row["radar"],
                           display_points=args.display_points,
                           seed=3103 + sum(frame_id.encode("utf-8")))
        output_rows.append({"frame_id": frame_id, "fault": fault, "file": filename})
        print(f"{frame_id} {fault}: {filename}", flush=True)
    (args.output_root / "index.html").write_text(_viewer_index(output_rows),
                                                  encoding="utf-8")
    print(f"Viewer: {args.output_root / 'index.html'}", flush=True)


if __name__ == "__main__":
    main()

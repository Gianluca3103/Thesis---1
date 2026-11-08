"""Interactive full-voxel comparison of radar support and the fault-only gate.

Every active voxel is encoded in each HTML page. The LiDAR/radar overlays may
be sampled for display, but the gate masks are complete and use the same
fault_only_support function as Stage II training and inference.
"""

from __future__ import annotations

import argparse
import base64
import html
import json
from pathlib import Path
import random

import numpy as np

from models.radar_lidar_crossmodal_v1.hierarchical_gate import missing_target, quantize
from models.radar_lidar_crossmodal_v1.pipeline_data import paired_dataset
from models.radar_lidar_crossmodal_v1.radar_pruner import unique_radar_voxel_centers
from models.radar_lidar_crossmodal_v1.radar_pruner_cache import (
    RadarSupportCache, identity as support_identity)
from models.radar_lidar_stage2.fault_data import load_faulty
from models.radar_lidar_stage2.fault_only_gate import fault_only_support
from models.radar_lidar_stage2.fault_region import FaultRegionCache, cache_identity
from models.radar_lidar_stage2.mild_fault_data import MILD_FOG2_TYPES, MILD_FOG2_VERSION
from models.radar_lidar_stage2.sparse_geometry import GRID
from scripts.compare_radar_support_rules import _sample, _viewer_html, _xyz, flat_indices


OBSERVED_BIT = 1 << 2


def candidate_payload(raw: np.ndarray, active: np.ndarray,
                      observed_index: np.ndarray) -> tuple[str, int]:
    """Pack two exact masks and the observed-LiDAR bit for the 3D canvas."""
    if (raw.shape != GRID.shape_zyx or active.shape != GRID.shape_zyx or
            raw.dtype != np.bool_ or active.dtype != np.bool_ or
            np.any(active & ~raw)):
        raise ValueError("Fault-only active cells must be a subset of radar support")
    candidate_index = np.flatnonzero(raw.ravel())
    membership = np.ones(len(candidate_index), dtype=np.uint32)
    membership |= active.ravel()[candidate_index].astype(np.uint32) << 1
    membership |= (np.isin(candidate_index, observed_index,
                            assume_unique=True).astype(np.uint32) << 2)
    packed = np.empty(len(candidate_index),
                      dtype=np.dtype([("index", "<u4"), ("membership", "<u4")]))
    packed["index"] = candidate_index
    packed["membership"] = membership
    return base64.b64encode(packed.tobytes()).decode("ascii"), len(candidate_index)


def render_frame(path: Path, *, frame_id: str, fault: str, region: dict,
                 raw: np.ndarray, active: np.ndarray, clean, faulty, radar,
                 display_points: int, seed: int) -> dict:
    observed = quantize(faulty, 1)
    missing = missing_target(clean[:, :3], faulty)
    observed_index = flat_indices(observed)
    missing_index = flat_indices(missing)
    packed, count = candidate_payload(raw, active, observed_index)
    rng = np.random.default_rng(seed)

    def rule(mask: np.ndarray, label: str) -> dict:
        flat = mask.ravel()
        supported = int(mask.sum())
        additions = supported - int(flat[observed_index].sum())
        eligible = int(flat[missing_index].sum())
        return {"label": label, "raw_cells": supported,
                "candidates": additions,
                "missing_recall": round(100 * eligible / max(len(missing), 1), 2)}

    detail = ("; fog touched full scan, so unchanged" if region["kind"] == "full_scan"
              else "; reliable sector removed")
    payload = {"frame_id": frame_id, "fault": fault,
               "rules": [rule(raw, "Original radar gate: ≥12 voxels in 1.6 m"),
                         rule(active, "Fault-only active region" + detail)],
               "candidate_count": count, "observed_bit": OBSERVED_BIT,
               "packed_candidates": packed,
               "grid": {"minimum": GRID.minimum_xyz, "size": GRID.voxel_xyz,
                        "shape": GRID.shape_zyx},
               "faulty": np.round(_sample(_xyz(observed), display_points, rng), 2).tolist(),
               "missing": np.round(_sample(_xyz(missing), display_points, rng), 2).tolist(),
               "radar": np.round(_sample(unique_radar_voxel_centers(radar),
                                          display_points, rng), 2).tolist()}
    page = _viewer_html(payload)
    page = page.replace("Complete radar geometry gates", "Stage II fault-only active region")
    page = page.replace("No model prediction or injector footprint is used.",
                        "The fault-only rule uses the known synthetic injector footprint. No model predictions are shown.")
    page = page.replace("box.checked=i===0", "box.checked=i===1")
    page = page.replace("17 radar support rules", "2 active-region rules")
    path.write_text(page, encoding="utf-8")
    return {"frame_id": frame_id, "fault": fault, "file": path.name,
            "radar_cells": int(raw.sum()), "fault_only_cells": int(active.sum()),
            "radar_addition_candidates": payload["rules"][0]["candidates"],
            "fault_only_addition_candidates": payload["rules"][1]["candidates"]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vod-root", type=Path, required=True)
    parser.add_argument("--data-cache-root", type=Path, required=True)
    parser.add_argument("--support-cache-root", type=Path, required=True)
    parser.add_argument("--faulty-root", type=Path, required=True)
    parser.add_argument("--fault-region-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "val", "test"), default="val")
    parser.add_argument("--radar-variant", default="radar_20frames_verified_doppler_radial")
    parser.add_argument("--frame-id", action="append")
    parser.add_argument("--per-fault", type=int, default=2)
    parser.add_argument("--display-points", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=3103)
    args = parser.parse_args()
    if args.per_fault < 1 or args.display_points < 1:
        parser.error("--per-fault and --display-points must be positive")
    dataset = paired_dataset(args.vod_root, args.split, args.radar_variant,
                             include_clean=True, data_cache_root=args.data_cache_root)
    ids = [frame.frame_id for frame in dataset.frames]
    manifest_path = args.faulty_root / args.split / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (manifest.get("version") != MILD_FOG2_VERSION or
            manifest.get("split") != args.split or
            manifest.get("radar_variant") != args.radar_variant or
            manifest.get("frame_ids") != ids):
        raise ValueError("Mild-fault cache differs from fixed frames")
    support = RadarSupportCache(
        args.support_cache_root / args.split,
        support_identity(args.split, ids, args.radar_variant,
                         data_cache_root=args.data_cache_root,
                         radius_m=1.6, min_radar_voxels=12))
    support.check_complete()
    regions = FaultRegionCache(
        args.fault_region_root / args.split,
        cache_identity(args.split, ids, manifest_path,
                       args.data_cache_root / args.split / "manifest.json"))
    if regions.regions is None:
        raise FileNotFoundError("Injector footprints are not built")
    if args.frame_id:
        unknown = set(args.frame_id) - set(ids)
        if unknown:
            raise ValueError(f"Unknown {args.split} frame IDs: {sorted(unknown)}")
        selected = list(dict.fromkeys(args.frame_id))
    else:
        rng = random.Random(args.seed)
        selected = []
        for fault in MILD_FOG2_TYPES:
            candidates = [frame_id for frame_id in ids
                          if manifest["frame_faults"][frame_id] == fault]
            selected.extend(rng.sample(candidates,
                                       min(args.per_fault, len(candidates))))
    lookup = {frame.frame_id: index for index, frame in enumerate(dataset.frames)}
    args.output_root.mkdir(parents=True, exist_ok=True)
    records = []
    for frame_id in selected:
        row = dataset[lookup[frame_id]]
        raw = support.load(frame_id)
        region = regions.get(frame_id)
        active = fault_only_support(raw, region)
        record = render_frame(
            args.output_root / f"{frame_id}_{region['fault_type']}.html",
            frame_id=frame_id, fault=region["fault_type"], region=region,
            raw=raw, active=active, clean=row["clean_lidar"],
            faulty=load_faulty(args.faulty_root, args.split, frame_id),
            radar=row["radar"], display_points=args.display_points,
            seed=args.seed + int(frame_id))
        records.append(record)
        print(f"{frame_id} {record['fault']}: "
              f"{record['radar_cells']:,} radar cells → "
              f"{record['fault_only_cells']:,} fault-only cells", flush=True)
    links = "".join(
        f'<li><a href="{html.escape(record["file"], quote=True)}">'
        f'{html.escape(record["frame_id"])} · {html.escape(record["fault"])}'
        f'</a> — {record["fault_only_cells"]:,} / '
        f'{record["radar_cells"]:,} cells</li>' for record in records)
    index = ("<!doctype html><html lang='en'><meta charset='utf-8'>"
             "<title>Stage II fault-only active region</title>"
             "<style>body{background:#101820;color:#eaf3f8;font:16px system-ui;"
             "padding:24px}a{color:#9dcafb}li{margin:10px}</style>"
             "<h1>Stage II fault-only active region</h1>"
             "<p>Each frame toggles complete radar and fault-only voxel masks. "
             "No model checkpoint is needed. The fault footprint is known "
             "from the synthetic injector.</p><ul>" + links + "</ul></html>")
    (args.output_root / "index.html").write_text(index, encoding="utf-8")
    print(f"Viewer: {args.output_root / 'index.html'}", flush=True)


if __name__ == "__main__":
    main()

"""Probe Stage-I candidate coverage and teacher versus student conditioning.

The teacher has no fine-voxel decoder. For a diagnostic only, its clean-LiDAR-
conditioned bottleneck is inserted into the *same* trained radar-only surface
head used by the student. This is not an independently trained teacher map or
an inference path: the teacher uses clean LiDAR and the head was trained on
student features. Oracle candidate recall ignores confidence but still uses
the checkpoint's learned 64 proposal locations.
"""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
import random

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import Subset
from tqdm import tqdm

from models.radar_lidar_stage1.data import VoDStage1Dataset, collate_stage1
from models.radar_lidar_stage1.radar_fov import (
    RADAR_FOV_SPEC, RADAR_HALF_FOV_DEG, RADAR_MAX_RANGE_M)
from models.radar_lidar_stage1.sparse import encode_keys, voxelize
from models.radar_lidar_crossmodal_v1.encoder import GRID
from models.radar_lidar_crossmodal_v1.fine_association import load_fine_association
from models.radar_lidar_crossmodal_v1.fixed_frame_cache import FixedFrameCache
from models.radar_lidar_crossmodal_v1.pointer_voxels import merge_surface_pointers
from models.radar_lidar_crossmodal_v1.train_teacher import load_teacher
from scripts.compare_crossmodal_surface import _write_html
from scripts.compare_fine_association_edge import (
    _audit_cloud, _metrics, _near_radar_voxel)


def _source_teacher(fine_saved: dict, override: Path | None) -> Path:
    if override is not None:
        return override
    student_path = Path(fine_saved["source_student_checkpoint"])
    if not student_path.is_file():
        raise FileNotFoundError(
            f"Source student missing: {student_path}; pass --teacher-checkpoint")
    source = torch.load(student_path, map_location="cpu", weights_only=False)
    teacher_path = Path(source["teacher_checkpoint"])
    if not teacher_path.is_file():
        raise FileNotFoundError(
            f"Source teacher missing: {teacher_path}; pass --teacher-checkpoint")
    return teacher_path


def _teacher_features_on_student_sites(teacher_sites, student_sites):
    if (teacher_sites.stride != student_sites.stride or
            teacher_sites.shape_zyx != student_sites.shape_zyx):
        raise ValueError("Teacher and student bottleneck grids differ")
    shape = student_sites.shape_zyx
    teacher_keys = encode_keys(teacher_sites.coords, shape)
    student_keys = encode_keys(student_sites.coords, shape)
    sorted_keys, order = torch.sort(teacher_keys)
    if not torch.equal(sorted_keys, torch.sort(student_keys).values):
        raise ValueError("Teacher and student radar bottleneck sites differ")
    positions = torch.searchsorted(sorted_keys, student_keys)
    return teacher_sites.features[order[positions]]


def _scored_field(surface, clean_coords, anchors, radius_m, threshold):
    field = merge_surface_pointers(surface)
    coords = field.coords.cpu()
    xyz = GRID.centers_xyz(coords).numpy()
    scores = field.confidence.cpu().numpy()
    supported = _near_radar_voxel(xyz, anchors, radius_m)
    scoped_coords = coords[torch.from_numpy(supported)]
    scoped_scores = scores[supported]
    selected = scoped_coords[torch.from_numpy(scoped_scores >= threshold)]
    oracle = _metrics(scoped_coords, scoped_scores, clean_coords, 0.0)
    chosen = _metrics(scoped_coords, scoped_scores, clean_coords, threshold)
    return {"xyz": xyz, "score": scores, "selected": selected,
            "oracle": oracle, "thresholded": chosen,
            "unique_candidates_full": len(coords),
            "unique_candidates_in_scope": len(scoped_coords)}


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vod-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True,
                        help="Trained 64-slot fine-association checkpoint")
    parser.add_argument("--teacher-checkpoint", type=Path,
                        help="Override the teacher recorded in the source student")
    parser.add_argument("--data-cache-root", type=Path)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "val"), default="val")
    parser.add_argument("--frame-id", nargs="+")
    parser.add_argument("--limit", type=int, default=3)
    parser.add_argument("--random-seed", type=int, default=3103)
    parser.add_argument("--threshold", type=float, default=0.6)
    parser.add_argument("--scope-m", type=float, choices=(1.0, 1.6), default=1.0)
    parser.add_argument("--top-k", type=int, choices=(32, 48, 64), default=64)
    parser.add_argument("--metrics-only", action="store_true",
                        help="Score frames without writing large HTML viewers")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.limit < 1 or not 0 < args.threshold < 1:
        parser.error("--limit must be positive and --threshold must lie in (0,1)")

    student, saved = load_fine_association(args.checkpoint, args.device)
    if saved.get("lidar_fov") != RADAR_FOV_SPEC:
        raise ValueError("Checkpoint and clean-LiDAR forward sector differ")
    if args.top_k > student.config.proposals_per_anchor:
        parser.error("--top-k exceeds the checkpoint's trained slots")
    teacher_path = _source_teacher(saved, args.teacher_checkpoint)
    teacher, teacher_saved = load_teacher(teacher_path, args.device)
    if (teacher_saved["radar_variant"] != saved["radar_variant"] or
            teacher_saved.get("lidar_fov") != saved.get("lidar_fov")):
        raise ValueError("Teacher and fine-association checkpoint use different data")
    teacher.eval()
    student.eval()

    full_dataset = VoDStage1Dataset(
        args.vod_root, args.split, radar_variant=saved["radar_variant"],
        include_clean=True,
        lidar_radar_fov_deg=RADAR_HALF_FOV_DEG,
        lidar_radar_max_range_m=RADAR_MAX_RANGE_M)
    if args.frame_id:
        by_id = {frame.frame_id: index for index, frame in enumerate(full_dataset.frames)}
        if missing := set(args.frame_id) - set(by_id):
            parser.error(f"Frame IDs absent from {args.split}: {sorted(missing)}")
        indices = [by_id[frame_id] for frame_id in args.frame_id]
    else:
        indices = random.Random(args.random_seed).sample(
            range(len(full_dataset)), min(args.limit, len(full_dataset)))
    if args.data_cache_root is not None:
        full_dataset = FixedFrameCache(args.data_cache_root / args.split,
                                       full_dataset, include_clean=True)
        full_dataset.check_complete()
    dataset = Subset(full_dataset, indices)
    args.output_root.mkdir(parents=True, exist_ok=True)
    reports = []
    for row in tqdm(dataset, total=len(dataset), desc="Stage I feature probe",
                    unit="frame", dynamic_ncols=True):
        batch = collate_stage1([row])
        radar = batch["radar"].to(args.device)
        radar_valid = batch["radar_valid"].to(args.device)
        clean = batch["clean_lidar"].to(args.device)
        clean_valid = batch["clean_lidar_valid"].to(args.device)

        teacher_output = teacher(radar, radar_valid, clean, clean_valid)
        coarse, levels = student.radar_encoder.forward_pyramid(radar, radar_valid)
        shared = coarse.replace_features(student.shared(coarse.features))
        student_sites = shared.replace_features(
            student.feature_norm(student.feature_head(shared.features)))
        teacher_features = _teacher_features_on_student_sites(
            teacher_output.fused, student_sites)
        teacher_sites = coarse.replace_features(teacher_features)
        teacher_surface = student.surface_head((*levels, teacher_sites))
        student_surface = student.surface_head((*levels, student_sites))
        if args.top_k < student.config.proposals_per_anchor:
            teacher_surface = teacher_surface.highest_confidence(args.top_k)
            student_surface = student_surface.highest_confidence(args.top_k)

        clean_coords, _, _, _ = voxelize(clean, clean_valid, GRID)
        clean_coords = clean_coords.cpu()
        clean_xyz = GRID.centers_xyz(clean_coords).numpy()
        anchors = student_surface.anchor_xyz.cpu().numpy()
        clean_scope = clean_coords[torch.from_numpy(
            _near_radar_voxel(clean_xyz, anchors, args.scope_m))]
        before = _scored_field(teacher_surface, clean_scope, anchors,
                               args.scope_m, args.threshold)
        after = _scored_field(student_surface, clean_scope, anchors,
                              args.scope_m, args.threshold)
        if len(student_sites.features):
            cosine = F.cosine_similarity(student_sites.features,
                                          teacher_features, dim=-1)
            feature_summary = {"mean_cosine": float(cosine.mean()),
                               "median_cosine": float(cosine.median()),
                               "mean_absolute_error": float((
                                   student_sites.features - teacher_features
                               ).abs().mean())}
        else:
            feature_summary = {"mean_cosine": None, "median_cosine": None,
                               "mean_absolute_error": None}
        report = {
            "frame_id": row["frame_id"], "split": args.split,
            "radar_variant": saved["radar_variant"], "epoch": saved["epoch"],
            "scope_m": args.scope_m, "threshold": args.threshold,
            "top_k_per_radar_voxel": args.top_k,
            "fine_checkpoint": str(args.checkpoint),
            "teacher_checkpoint": str(teacher_path),
            "teacher_requires_clean_lidar": True,
            "teacher_head_trained_on_student_features": True,
            "feature_comparison": feature_summary,
            "clean_voxels_in_scope": len(clean_scope),
            "radar_anchors": len(anchors),
            "teacher_conditioned": {key: value for key, value in before.items()
                                    if key not in ("xyz", "score", "selected")},
            "radar_only_distilled": {key: value for key, value in after.items()
                                     if key not in ("xyz", "score", "selected")},
        }
        stem = args.output_root / f"{row['frame_id']}_teacher_vs_radar_only"
        if not args.metrics_only:
            audit_clouds = [
                _audit_cloud("Teacher-conditioned audit (privileged)",
                             before["selected"], clean_scope,
                             before["thresholded"]["tp_at_0_2m"]),
                _audit_cloud("Radar-only audit (inference path)",
                             after["selected"], clean_scope,
                             after["thresholded"]["tp_at_0_2m"]),
            ]
            _write_html(
                stem.with_suffix(".html"), row["frame_id"], saved["epoch"],
                row["radar"][:, :3].numpy(), row["clean_lidar"][:, :3].numpy(),
                before["xyz"], before["score"], after["xyz"], after["score"],
                threshold=args.threshold, links=None, audit_clouds=audit_clouds,
                raw_label="Teacher-conditioned probe (uses clean LiDAR)",
                edge_label="Radar-only distilled prediction",
                page_title="teacher feature versus radar-only feature probe",
                description=("The same trained 64-slot head receives either the "
                             "clean-LiDAR-conditioned teacher bottleneck or the "
                             "radar-only student bottleneck. This is a feature-swap "
                             "diagnostic, not an independently trained teacher map. "
                             "The slider changes the two prediction panels only."),
                audit_note=("Fixed-threshold 0.2 m audits in the selected radar "
                            f"{args.scope_m:g} m region. Green: match; red: false "
                            "positive; blue: missed clean voxel. Oracle recall in "
                            "the JSON ignores confidence and measures candidate coverage."))
            stem.with_suffix(".json").write_text(
                json.dumps(report, indent=2), encoding="utf-8")
        reports.append(report)
        if not args.metrics_only:
            print(f"{row['frame_id']}: teacher oracle recall "
                  f"{before['oracle']['recall_at_0_2m']}; radar-only oracle recall "
                  f"{after['oracle']['recall_at_0_2m']} | {stem.with_suffix('.html')}",
                  flush=True)
    if not args.metrics_only:
        index = ("<!doctype html><meta charset='utf-8'><title>Stage I feature probe</title>"
                 "<style>body{font:16px system-ui;background:#101820;color:#edf4fa}"
                 "a{color:#88ceff}iframe{width:100%;height:88vh;border:1px solid #526474}</style>"
                 "<h1>Stage I teacher-conditioned versus radar-only feature probe</h1>"
                 "<p>Teacher uses clean LiDAR. Both maps use the same trained radar-only head.</p>"
                 "<ul>" + "".join(
                     f"<li><a target='viewer' href='{html.escape(r['frame_id'])}_teacher_vs_radar_only.html'>"
                     f"Frame {html.escape(r['frame_id'])}</a></li>" for r in reports) +
                 "</ul><iframe name='viewer' src='" +
                 html.escape(reports[0]["frame_id"]) + "_teacher_vs_radar_only.html'></iframe>")
        (args.output_root / "index.html").write_text(index, encoding="utf-8")
    aggregate = {}
    for name in ("teacher_conditioned", "radar_only_distilled"):
        aggregate[name] = {}
        for mode in ("oracle", "thresholded"):
            tp = sum(r[name][mode]["tp_at_0_2m"] for r in reports)
            observed = sum(r[name][mode]["observed"] for r in reports)
            predicted = sum(r[name][mode]["predicted"] for r in reports)
            aggregate[name][mode] = {
                "tp_at_0_2m": tp, "observed": observed, "predicted": predicted,
                "recall_at_0_2m": tp / observed if observed else None,
                "precision_at_0_2m": tp / predicted if predicted else None,
            }
    (args.output_root / "summary.json").write_text(
        json.dumps({"aggregate": aggregate, "frames": reports}, indent=2),
        encoding="utf-8")
    print(json.dumps(aggregate, indent=2), flush=True)
    if not args.metrics_only:
        print(f"Open: {args.output_root / 'index.html'}", flush=True)


if __name__ == "__main__":
    main()

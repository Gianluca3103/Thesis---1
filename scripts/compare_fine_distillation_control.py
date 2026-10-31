"""Held-out Stage-I predictions before/after source-student distillation.

Thresholds are selected on one half of validation frames and scored on the
disjoint half. Raw radar and distilled radar use no LiDAR as model input. The
teacher arm is privileged: it reads clean LiDAR and is a reference ceiling.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import asdict
import html
import json
from pathlib import Path
import random

import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from models.radar_lidar_stage1.sparse import voxelize
from models.radar_lidar_crossmodal_v1.encoder import GRID
from models.radar_lidar_crossmodal_v1.pipeline_data import (
    collate_stage1, paired_dataset)
from models.radar_lidar_crossmodal_v1.pointer_voxels import merge_surface_pointers
from models.radar_lidar_crossmodal_v1.surface_student import SurfaceStudentConfig
from models.radar_lidar_crossmodal_v1.train_teacher import load_teacher
from scripts.compare_crossmodal_surface import _write_html
from scripts.compare_fine_association_edge import (
    _audit_cloud, _metrics, _near_radar_voxel)
from scripts.train_fine_distillation_control import (
    ARMS, VERSION, MatchedHeads, sha256)


THRESHOLDS = (0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80,
              0.90, 0.95, 0.975, 0.99)
SCOPES = (1.0, 1.6)
COUNT_KEYS = ("predicted", "observed", "tp_exact", "tp_at_0_2m")


def _sum_counts(total: dict, row: dict) -> None:
    for key in COUNT_KEYS:
        total[key] += int(row[key])


def _rates(counts: dict) -> dict:
    predicted, observed = counts["predicted"], counts["observed"]
    result = dict(counts)
    for label, tp_key in (("exact", "tp_exact"),
                          ("at_0_2m", "tp_at_0_2m")):
        tp = counts[tp_key]
        result[f"precision_{label}"] = tp / predicted if predicted else None
        result[f"recall_{label}"] = tp / observed if observed else None
        result[f"iou_{label}"] = (
            tp / (predicted + observed - tp)
            if predicted + observed - tp else None)
        result[f"f1_{label}"] = (
            2 * tp / (predicted + observed) if predicted + observed else None)
    return result


def _field(surface, top_k: int):
    if top_k < surface.logits.shape[1]:
        surface = surface.highest_confidence(top_k)
    field = merge_surface_pointers(surface)
    coords = field.coords.cpu()
    return {"coords": coords, "xyz": GRID.centers_xyz(coords).numpy(),
            "scores": field.confidence.cpu().numpy(),
            "anchors": surface.anchor_xyz.cpu().numpy()}


def _scope(field, clean_coords, clean_xyz, radius):
    predicted_mask = _near_radar_voxel(field["xyz"], field["anchors"], radius)
    clean_mask = _near_radar_voxel(clean_xyz, field["anchors"], radius)
    return (field["coords"][torch.from_numpy(predicted_mask)],
            field["scores"][predicted_mask],
            clean_coords[torch.from_numpy(clean_mask)])


def _load(checkpoint: Path, device: str):
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if saved.get("version") != VERSION:
        raise ValueError("Expected a matched fine-distillation checkpoint")
    source_path = Path(saved["source_student"])
    teacher_path = Path(saved["teacher_checkpoint"])
    if (sha256(source_path) != saved["source_sha256"] or
            sha256(teacher_path) != saved["teacher_sha256"]):
        raise ValueError("Frozen source or teacher changed since training")
    source = torch.load(source_path, map_location="cpu", weights_only=False)
    teacher, teacher_saved = load_teacher(teacher_path, device)
    raw = dict(saved["config"])
    raw["context_radii_m"] = tuple(raw["context_radii_m"])
    config = SurfaceStudentConfig(**raw)
    if asdict(config) != saved["config"]:
        # JSON/list versus tuple is the only expected representation change.
        if {**asdict(config), "context_radii_m": list(config.context_radii_m)} != saved["config"]:
            raise ValueError("Checkpoint config differs from deserialized config")
    if (source["radar_variant"] != teacher_saved["radar_variant"] or
            source.get("lidar_fov") != teacher_saved.get("lidar_fov")):
        raise ValueError("Source and teacher use different data")
    model = MatchedHeads(config, source).to(device)
    model.heads.load_state_dict(saved["heads"], strict=True)
    model.eval()
    teacher.requires_grad_(False)
    teacher.eval()
    return model, teacher, source["radar_variant"], saved


def _viewer_pair(path: Path, row, saved, first_name, first,
                 second_name, second, clean_scope, threshold, scope_m):
    first_coords, first_scores, _ = _scope(first, clean_scope["all_coords"],
                                           clean_scope["all_xyz"], scope_m)
    second_coords, second_scores, _ = _scope(second, clean_scope["all_coords"],
                                             clean_scope["all_xyz"], scope_m)
    observed = clean_scope["by_radius"][scope_m]
    first_selected = first_coords[torch.from_numpy(first_scores >= threshold)]
    second_selected = second_coords[torch.from_numpy(second_scores >= threshold)]
    first_stats = _metrics(first_coords, first_scores, observed, threshold)
    second_stats = _metrics(second_coords, second_scores, observed, threshold)
    audits = [
        _audit_cloud(f"{first_name} audit", first_selected, observed,
                     first_stats["tp_at_0_2m"]),
        _audit_cloud(f"{second_name} audit", second_selected, observed,
                     second_stats["tp_at_0_2m"]),
    ]
    _write_html(
        path, row["frame_id"], saved["epoch"], row["radar"][:, :3].numpy(),
        row["clean_lidar"][:, :3].numpy(), first["xyz"], first["scores"],
        second["xyz"], second["scores"], threshold=threshold, links=None,
        audit_clouds=audits, raw_label=first_name, edge_label=second_name,
        page_title="matched conditioning heads",
        description=("All heads were trained independently from identical initial "
                     "weights with the same targets, radar pyramid, and schedule. "
                     "Clean LiDAR is shown as reference; only the teacher arm "
                     "receives it as model input. The slider changes prediction panels."),
        audit_note=(f"Audits use a fixed {threshold:.2f} threshold in the radar "
                    f"{scope_m:g} m region. Blue: missed clean voxel; green: "
                    "one-to-one 0.2 m match; red: unmatched prediction."))


@torch.no_grad()
def _pass(model, teacher, loader, device, *, top_k, thresholds,
          output_root=None, display_threshold=0.6, viewer_limit=0,
          saved=None):
    totals = defaultdict(lambda: {key: 0 for key in COUNT_KEYS})
    viewers = []
    for batch in tqdm(loader, desc="Distillation control validation",
                      unit="frame", dynamic_ncols=True):
        if len(batch["frame_id"]) != 1:
            raise ValueError("Evaluation requires batch size 1 for frame-local matching")
        radar = batch["radar"].to(device)
        radar_valid = batch["radar_valid"].to(device)
        clean = batch["clean_lidar"].to(device)
        clean_valid = batch["clean_lidar_valid"].to(device)
        outputs = model(radar, radar_valid, clean, clean_valid, teacher)
        fields = {name: _field(outputs[name].surface, top_k) for name in ARMS}
        clean_coords, _, _, _ = voxelize(clean, clean_valid, GRID)
        clean_coords = clean_coords.cpu()
        clean_xyz = GRID.centers_xyz(clean_coords).numpy()
        clean_scope = {"all_coords": clean_coords, "all_xyz": clean_xyz,
                       "by_radius": {}}
        for radius in SCOPES:
            # All three arms share the same active fine radar anchors.
            clean_scope["by_radius"][radius] = clean_coords[torch.from_numpy(
                _near_radar_voxel(clean_xyz,
                                  fields["raw_radar"]["anchors"], radius))]
            for name in ARMS:
                coords, scores, observed = _scope(
                    fields[name], clean_coords, clean_xyz, radius)
                _sum_counts(totals[(name, radius, "oracle")],
                            _metrics(coords, scores, observed, 0.0))
                for threshold in thresholds[name, radius]:
                    _sum_counts(totals[(name, radius, threshold)],
                                _metrics(coords, scores, observed, threshold))
        if output_root is not None and len(viewers) < viewer_limit:
            frame = batch["frame_id"][0]
            row = {"frame_id": frame,
                   "radar": batch["radar"][0, batch["radar_valid"][0]],
                   "clean_lidar": batch["clean_lidar"][0, batch["clean_lidar_valid"][0]]}
            raw_vs_distilled = output_root / f"{frame}_raw_vs_distilled.html"
            teacher_vs_distilled = output_root / f"{frame}_teacher_vs_distilled.html"
            _viewer_pair(raw_vs_distilled, row, saved, "Raw radar (before distillation)",
                         fields["raw_radar"], "Distilled radar (radar only)",
                         fields["distilled_radar"], clean_scope,
                         display_threshold, 1.0)
            _viewer_pair(teacher_vs_distilled, row, saved,
                         "Teacher (clean LiDAR + radar)", fields["teacher_conditioned"],
                         "Distilled radar (radar only)", fields["distilled_radar"],
                         clean_scope, display_threshold, 1.0)
            viewers.append({"frame": frame,
                            "raw_vs_distilled": raw_vs_distilled.name,
                            "teacher_vs_distilled": teacher_vs_distilled.name})
    return {key: _rates(value) for key, value in totals.items()}, viewers


def _index(path: Path, viewers: list[dict]) -> None:
    links = "".join(
        f"<li>Frame {html.escape(item['frame'])}: "
        f"<a target='viewer' href='{html.escape(item['raw_vs_distilled'])}'>"
        "raw radar versus distilled radar</a> · "
        f"<a target='viewer' href='{html.escape(item['teacher_vs_distilled'])}'>"
        "teacher versus distilled radar</a></li>" for item in viewers)
    document = ("<!doctype html><meta charset='utf-8'><title>Distillation control</title>"
                "<style>body{font:16px system-ui;background:#101820;color:#edf4fa}"
                "a{color:#88ceff}iframe{width:100%;height:86vh;border:1px solid #526474}</style>"
                "<h1>Matched Stage I prediction heads</h1>"
                "<p>Raw and distilled arms are radar only. The teacher arm uses clean LiDAR.</p>"
                f"<ul>{links}</ul><iframe name='viewer' src='"
                f"{html.escape(viewers[0]['raw_vs_distilled'])}'></iframe>")
    path.write_text(document, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vod-root", type=Path, required=True)
    parser.add_argument("--data-cache-root", type=Path)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=1296)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--top-k", type=int, choices=(32, 48, 64), default=64)
    parser.add_argument("--viewer-limit", type=int, default=5)
    parser.add_argument("--display-threshold", type=float, default=0.6)
    parser.add_argument("--seed", type=int, default=3103)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if (args.limit < 4 or args.num_workers < 0 or args.viewer_limit < 0
            or not 0 < args.display_threshold < 1):
        parser.error("Invalid limit, worker count, viewer count, or threshold")
    model, teacher, variant, saved = _load(args.checkpoint, args.device)
    if args.top_k > model.config.proposals_per_anchor:
        parser.error("Top-K exceeds the trained 64 slots")
    dataset = paired_dataset(args.vod_root, "val", variant,
                             data_cache_root=args.data_cache_root)
    permutation = list(range(len(dataset)))
    random.Random(args.seed).shuffle(permutation)
    chosen = permutation[:min(args.limit, len(permutation))]
    midpoint = len(chosen) // 2
    calibration, held_out = Subset(dataset, chosen[:midpoint]), Subset(dataset, chosen[midpoint:])
    def loader(part):
        return DataLoader(part, batch_size=1, shuffle=False,
                          num_workers=args.num_workers,
                          collate_fn=collate_stage1,
                          pin_memory=args.device.startswith("cuda"))
    args.output_root.mkdir(parents=True, exist_ok=True)
    all_thresholds = {(name, radius): THRESHOLDS for name in ARMS for radius in SCOPES}
    calibration_rows, _ = _pass(model, teacher, loader(calibration), args.device,
                                 top_k=args.top_k, thresholds=all_thresholds)
    chosen_thresholds = {}
    for name in ARMS:
        for radius in SCOPES:
            best = max(THRESHOLDS, key=lambda threshold: (
                calibration_rows[name, radius, threshold]["f1_at_0_2m"] or 0.0,
                -threshold))
            chosen_thresholds[name, radius] = best
    held_thresholds = {(name, radius):
                       tuple(sorted({chosen_thresholds[name, radius],
                                     args.display_threshold}))
                       for name in ARMS for radius in SCOPES}
    held_rows, viewers = _pass(
        model, teacher, loader(held_out), args.device,
        top_k=args.top_k, thresholds=held_thresholds,
        output_root=args.output_root, display_threshold=args.display_threshold,
        viewer_limit=args.viewer_limit, saved=saved)
    result = {"checkpoint": str(args.checkpoint), "epoch": saved["epoch"],
              "radar_variant": variant, "top_k": args.top_k,
              "calibration_frames": len(calibration),
              "held_out_frames": len(held_out),
              "selection_metric": "f1_at_0_2m",
              "scope_note": "Clean and predicted voxels near active fine radar anchors",
              "teacher_requires_clean_lidar": True,
              "source_student_training_included_occupancy_loss": True,
              "results": {}, "viewers": viewers}
    for name in ARMS:
        result["results"][name] = {}
        for radius in SCOPES:
            threshold = chosen_thresholds[name, radius]
            result["results"][name][f"radar_{radius:g}m"] = {
                "selected_threshold": threshold,
                "calibration_at_selected": calibration_rows[name, radius, threshold],
                "held_out_at_selected": held_rows[name, radius, threshold],
                "held_out_at_display_threshold": held_rows[
                    name, radius, args.display_threshold],
                "held_out_oracle_all_candidates": held_rows[name, radius, "oracle"],
            }
    (args.output_root / "comparison_summary.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8")
    if viewers:
        _index(args.output_root / "index.html", viewers)
    print("Held-out radar 1 m results (each threshold calibrated separately):")
    def display(value):
        return f"{value:.4f}" if value is not None else "undefined"
    for name in ARMS:
        row = result["results"][name]["radar_1m"]
        selected = row["held_out_at_selected"]
        oracle = row["held_out_oracle_all_candidates"]
        print(f"{name:<21} threshold {row['selected_threshold']:.3f} "
              f"P/R/F1@0.2m {display(selected['precision_at_0_2m'])}/"
              f"{display(selected['recall_at_0_2m'])}/"
              f"{display(selected['f1_at_0_2m'])}; "
              f"oracle candidate recall {display(oracle['recall_at_0_2m'])}",
              flush=True)
    print(f"Summary: {args.output_root / 'comparison_summary.json'}")
    if viewers:
        print(f"Open: {args.output_root / 'index.html'}")


if __name__ == "__main__":
    main()

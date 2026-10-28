"""Sweep Stage-I/II thresholds and diagnose Stage-II false-positive additions.

Stage I is scored in radar support within the Stage-II ROI. Stage II is scored
over the full ROI after preserving faulty returns and adding reconstruction.
"stage1_active" scores Stage I only on its generated, radar-supported cells.
It does not penalize clean LiDAR voxels outside those cells; Chamfer is undefined.
Diffusion runs once per frame at the lowest occupancy threshold; higher
thresholds filter the same decoded points. Ray categories use XYZ-only proxy
matching because the paired scan contains no explicit beam IDs. Per-frame
records support resume.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

# Import ME before the model modules. Their optional-import guards otherwise
# hide the original ImportError and leave the decoder with ME = None.
if __name__ == "__main__":
    try:
        import MinkowskiEngine as _me_preload  # noqa: F401
    except ImportError as error:
        raise RuntimeError(
            "The Stage-II Python environment cannot import MinkowskiEngine; "
            "the original import error follows"
        ) from error

import numpy as np
import torch
from tqdm import tqdm

from models.radar_lidar_crossmodal_v1.pipeline_data import paired_dataset
from models.radar_lidar_stage2.completion_eval import (
    FP_CATEGORIES, classify_addition_false_positives, overlap_counts,
    rates, restrict_to_cells, score, scoped_cloud, voxel_centers,
)
from models.radar_lidar_stage2.fault_data import (
    FAULT_NAMES, FAULT_PLAN, FAULT_POLICY_VERSION, SevereFogCache, severe_fault,
)
from models.radar_lidar_stage2.sparse_geometry import GRID, select_allowed
from models.radar_lidar_stage2.sparse_residual import cosine_alpha_bars
from models.radar_lidar_stage2.stage2_cache import Stage2FixedCache, cache_identity
from models.radar_lidar_stage2.stage2_io import (
    load_stage1, load_stage2, sha256, stage1_frame,
)


VERSION = "stage1_stage2_completion_sweep_v2_ray_diagnostic"
DEFAULT_THRESHOLDS = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,
                      0.95, 0.975, 0.99)
METRICS = ("iou_exact", "iou_at_0_2m", "f1_exact", "f1_at_0_2m",
           "precision_exact", "precision_at_0_2m", "recall_exact",
           "recall_at_0_2m", "f0_5_exact", "f0_5_at_0_2m",
           "chamfer_squared_l2")
COUNT_FIELDS = ("predicted", "observed", "tp_exact", "tp_at_0_2m")
FP_FIELDS = ("addition_points", "true_positive_additions", "false_positive_additions",
             "no_retained_return_false_positives", *FP_CATEGORIES)
ACTIVE_FLOOR = 1e-30


def _atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    temporary.replace(path)


def _source_hash() -> str:
    digest = hashlib.sha256()
    for path in (Path(__file__), Path(__file__).parent.parent / "models"
                 / "radar_lidar_stage2" / "completion_eval.py"):
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _new_total() -> dict:
    return {**{name: 0 for name in COUNT_FIELDS}, "chamfer_sum": 0.0,
            "chamfer_defined_frames": 0, "frames": 0,
            "paired_chamfer_model_sum": 0.0,
            "paired_chamfer_faulty_sum": 0.0,
            "paired_chamfer_frames": 0}


def _add(total: dict, item: dict, baseline: dict | None = None) -> None:
    total["frames"] += 1
    for name in COUNT_FIELDS:
        total[name] += item[name]
    cd = item["chamfer_squared_l2"]
    if cd is not None:
        total["chamfer_sum"] += cd
        total["chamfer_defined_frames"] += 1
    if baseline is not None and cd is not None and baseline["chamfer_squared_l2"] is not None:
        total["paired_chamfer_model_sum"] += cd
        total["paired_chamfer_faulty_sum"] += baseline["chamfer_squared_l2"]
        total["paired_chamfer_frames"] += 1


def _finish(total: dict) -> dict:
    counts = {name: total[name] for name in COUNT_FIELDS}
    return {"frames": total["frames"], **counts, **rates(counts),
            "chamfer_squared_l2": (
                total["chamfer_sum"] / total["chamfer_defined_frames"]
                if total["chamfer_defined_frames"] else None),
            "chamfer_defined_frames": total["chamfer_defined_frames"]}


def _improvement(model: dict, faulty: dict, total: dict) -> dict:
    result = {}
    for metric in ("iou_exact", "iou_at_0_2m", "f1_exact", "f1_at_0_2m"):
        a, b = model[metric], faulty[metric]
        delta = a - b if a is not None and b is not None else None
        result[f"{metric}_gain"] = delta
        result[f"{metric}_relative_gain_pct"] = (
            100 * delta / b if delta is not None and b > 0 else None)
    pairs = total["paired_chamfer_frames"]
    faulty_cd = total["paired_chamfer_faulty_sum"] / pairs if pairs else None
    model_cd = total["paired_chamfer_model_sum"] / pairs if pairs else None
    reduction = faulty_cd - model_cd if pairs else None
    result.update({"chamfer_paired_frames": pairs,
                   "chamfer_faulty_paired_mean": faulty_cd,
                   "chamfer_model_paired_mean": model_cd,
                   "chamfer_squared_l2_reduction": reduction,
                   "chamfer_relative_reduction_pct": (
                       100 * reduction / faulty_cd
                       if faulty_cd is not None and faulty_cd > 0 else None)})
    return result


def _summarize_false_positives(records: list[dict],
                               thresholds: tuple[float, ...]) -> dict:
    scopes = {}
    for fault in ("overall", *FAULT_NAMES):
        selected = [record for record in records
                    if fault == "overall" or record["fault_type"] == fault]
        rows = []
        for index, threshold in enumerate(thresholds):
            counts = {key: 0 for key in FP_FIELDS}
            for frame in selected:
                for key in FP_FIELDS:
                    counts[key] += frame["stage2"]["false_positive_rows"][index][key]
            if counts["addition_points"] != (counts["true_positive_additions"]
                                              + counts["false_positive_additions"]):
                raise AssertionError("Addition diagnostic counts do not partition outputs")
            if counts["false_positive_additions"] != sum(
                    counts[key] for key in FP_CATEGORIES):
                raise AssertionError("False-positive categories do not partition outputs")
            rows.append({"threshold": threshold, **counts,
                         "false_positive_fraction": (
                             counts["false_positive_additions"] / counts["addition_points"]
                             if counts["addition_points"] else None),
                         "no_retained_return_fp_fraction": (
                             counts["no_retained_return_false_positives"]
                             / counts["false_positive_additions"]
                             if counts["false_positive_additions"] else None)})
        allowed = sum(frame["stage2_allowed_cells"] for frame in selected)
        clean = sum(frame["stage2_clean_cells"] for frame in selected)
        covered = sum(frame["stage2_clean_cells_in_gate"] for frame in selected)
        scopes[fault] = {"frames": len(selected), "rows": rows,
                         "gate": {"allowed_cells": allowed, "clean_cells": clean,
                                  "clean_cells_in_gate": covered,
                                  "clean_cell_coverage": covered / clean if clean else None,
                                  "clean_return_fraction_of_allowed": (
                                      covered / allowed if allowed else None)}}
    return scopes


def _summarize(records: list[dict], thresholds: tuple[float, ...],
               selection_metric: str,
               stages: tuple[str, ...] = ("stage1", "stage2")) -> dict:
    result = {}
    for stage in stages:
        scopes = {}
        for fault in ("overall", *FAULT_NAMES):
            selected = [record for record in records
                        if fault == "overall" or record["fault_type"] == fault]
            baseline_total = _new_total()
            row_totals = [_new_total() for _ in thresholds]
            for frame in selected:
                baseline = frame[stage]["faulty"]
                _add(baseline_total, baseline)
                for total, row in zip(row_totals, frame[stage]["rows"]):
                    _add(total, row, baseline)
            baseline = _finish(baseline_total)
            rows = []
            for threshold, total in zip(thresholds, row_totals):
                metric = _finish(total)
                rows.append({"threshold": threshold, **metric,
                             **_improvement(metric, baseline, total)})
            scopes[fault] = {"faulty_baseline": baseline, "rows": rows}
        metric_name = ("iou_exact" if stage == "stage1_active"
                       and selection_metric == "chamfer_squared_l2"
                       else selection_metric)
        valid = [row for row in scopes["overall"]["rows"]
                 if row[metric_name] is not None
                 and (metric_name != "chamfer_squared_l2"
                      or row["chamfer_defined_frames"] == row["frames"])]
        if not valid:
            raise ValueError(f"No valid {stage} values for {metric_name}")
        if metric_name == "chamfer_squared_l2":
            best = min(valid, key=lambda row: (row[metric_name], -row["threshold"]))
        else:
            best = max(valid, key=lambda row: (row[metric_name], row["threshold"]))
        result[stage] = {"selected_threshold": best["threshold"],
                         "selection_metric": metric_name, "scopes": scopes}
    return result


def _active_score(predicted_zyx: np.ndarray, observed_zyx: np.ndarray) -> dict:
    counts = overlap_counts(predicted_zyx, observed_zyx)
    return {**counts, **rates(counts), "chamfer_squared_l2": None}


@torch.no_grad()
def _frame_scores(frame: dict, frame_id: str, *, stage1, stage2,
                  stage2_cache: Stage2FixedCache | None, fog_cache: SevereFogCache,
                  alpha_bars: torch.Tensor, radius_m: float, gate_threshold: float,
                  thresholds: tuple[float, ...], sampling_steps: int,
                  random_seed: int, fault_seed: int, forced_fault: str | None,
                  ray_angle_deg: float, ray_lateral_m: float,
                  device: str, stage1_only: bool = False) -> dict:
    clean = frame["clean_lidar"][:, :3].numpy()
    faulty_tensor, fault_type = severe_fault(
        frame["clean_lidar"], frame_id, 0, fog_cache,
        seed=fault_seed, fault_type=forced_fault)
    faulty = faulty_tensor.numpy()
    radar_xyz = frame["radar"][:, :3].numpy()
    live = stage1_frame(stage1, frame["radar"].to(device), radius_m)
    stage1_coords, stage1_scores = select_allowed(
        live["confidence_coords"], live["confidence_probabilities"],
        min(thresholds))
    stage1_coords = stage1_coords[:, 1:].cpu().numpy()
    stage1_scores = stage1_scores.cpu().numpy().reshape(-1)
    active_coords, active_scores = select_allowed(
        live["confidence_coords"], live["confidence_probabilities"],
        ACTIVE_FLOOR)
    active_coords = active_coords[:, 1:].cpu().numpy()
    active_scores = active_scores.cpu().numpy().reshape(-1)
    clean1_xyz, clean1_zyx = scoped_cloud(clean, radar_xyz=radar_xyz,
                                           radius_m=radius_m)
    faulty1_xyz, faulty1_zyx = scoped_cloud(faulty, radar_xyz=radar_xyz,
                                             radius_m=radius_m)
    stage1_baseline = score(faulty1_xyz, faulty1_zyx,
                            clean1_xyz, clean1_zyx)
    stage1_rows = []
    for threshold in thresholds:
        coords = stage1_coords[stage1_scores >= threshold]
        stage1_rows.append({"threshold": threshold,
                            **score(voxel_centers(coords), coords,
                                    clean1_xyz, clean1_zyx)})

    clean_active = restrict_to_cells(clean1_zyx, active_coords)
    faulty_active = restrict_to_cells(faulty1_zyx, active_coords)
    active_baseline = _active_score(faulty_active, clean_active)
    active_rows = []
    for threshold in thresholds:
        coords = active_coords[active_scores >= threshold]
        active_rows.append({"threshold": threshold,
                            **_active_score(coords, clean_active)})
    if stage1_only:
        return {"frame_id": frame_id, "fault_type": fault_type,
                "stage1": {"faulty": stage1_baseline, "rows": stage1_rows},
                "stage1_active": {"faulty": active_baseline, "rows": active_rows},
                "stage1_active_cells": len(active_coords)}

    clean2_xyz, clean2_zyx = scoped_cloud(clean)
    faulty2_xyz, faulty2_zyx = scoped_cloud(faulty)
    stage2_baseline = score(faulty2_xyz, faulty2_zyx,
                            clean2_xyz, clean2_zyx)
    stage2_output = (stage2_cache.load(frame_id, device)[0]
                     if stage2_cache is not None else live)
    conditions = stage2.conditions(stage2_output, gate_threshold)
    allowed_zyx = conditions.hierarchy[0][:, 1:].cpu().numpy()
    allowed_keys = np.ravel_multi_index(allowed_zyx.T, GRID.shape_zyx)
    clean_keys = np.ravel_multi_index(clean2_zyx.T, GRID.shape_zyx)
    clean_in_gate = int(np.isin(clean_keys, allowed_keys).sum())
    if len(conditions.hierarchy[0]):
        seed = int.from_bytes(hashlib.sha256(
            f"{random_seed}:{frame_id}".encode()).digest()[:8], "little")
        torch.manual_seed(seed)
        prediction = stage2.sample(
            faulty_tensor.to(device), conditions, alpha_bars,
            steps=sampling_steps, occupancy_threshold=min(thresholds),
            missing_only=True)
        additions = prediction["points_xyz"].cpu().numpy()
        addition_scores = prediction["probabilities"].cpu().numpy()
    else:
        additions = np.empty((0, 3), dtype=np.float32)
        addition_scores = np.empty(0, dtype=np.float32)
    fp_labels, no_retained = classify_addition_false_positives(
        additions, clean, faulty,
        angular_tolerance_deg=ray_angle_deg,
        lateral_tolerance_m=ray_lateral_m)
    stage2_rows = []
    fp_rows = []
    for threshold in thresholds:
        selected_mask = addition_scores >= threshold
        selected = additions[selected_mask]
        completed = np.concatenate((faulty2_xyz, selected), axis=0)
        completed_xyz, completed_zyx = scoped_cloud(completed)
        stage2_rows.append({"threshold": threshold,
                            **score(completed_xyz, completed_zyx,
                                    clean2_xyz, clean2_zyx)})
        labels = fp_labels[selected_mask]
        fp_rows.append({"threshold": threshold,
                        "addition_points": len(labels),
                        "true_positive_additions": int(np.count_nonzero(labels == "true_positive")),
                        "false_positive_additions": int(np.count_nonzero(labels != "true_positive")),
                        "no_retained_return_false_positives": int(np.count_nonzero(
                            no_retained[selected_mask] & (labels != "true_positive"))),
                        **{name: int(np.count_nonzero(labels == name))
                           for name in FP_CATEGORIES}})
    return {"frame_id": frame_id, "fault_type": fault_type,
            "stage1": {"faulty": stage1_baseline, "rows": stage1_rows},
            "stage1_active": {"faulty": active_baseline, "rows": active_rows},
            "stage2": {"faulty": stage2_baseline, "rows": stage2_rows,
                       "false_positive_rows": fp_rows},
            "stage1_active_cells": len(active_coords),
            "stage2_allowed_cells": len(conditions.hierarchy[0]),
            "stage2_clean_cells": len(clean2_zyx),
            "stage2_clean_cells_in_gate": clean_in_gate,
            "stage2_lowest_threshold_additions": len(additions)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vod-root", type=Path, required=True)
    parser.add_argument("--data-cache-root", type=Path)
    parser.add_argument("--stage2-cache-root", type=Path)
    parser.add_argument("--fault-cache-root", type=Path, required=True)
    parser.add_argument("--stage1-checkpoint", type=Path, required=True)
    parser.add_argument("--stage2-checkpoint", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--radar-variant", default="radar_5frames")
    parser.add_argument("--split", choices=("train", "val"), default="val")
    parser.add_argument("--fault-type", choices=("auto", *FAULT_NAMES), default="auto")
    parser.add_argument("--thresholds", nargs="+", type=float,
                        default=DEFAULT_THRESHOLDS)
    parser.add_argument("--select-by", choices=METRICS, default="iou_exact")
    parser.add_argument("--sampling-steps", type=int, default=64)
    parser.add_argument("--ray-angle-deg", type=float, default=0.15,
                        help="Maximum angular separation for the XYZ-only ray proxy")
    parser.add_argument("--ray-lateral-m", type=float, default=0.15,
                        help="Maximum perpendicular distance to a measured ray")
    parser.add_argument("--random-seed", type=int, default=3103)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--stage1-only", action="store_true",
                        help="Skip Stage-II diffusion; score Stage I only")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if (args.device != "cuda" or not torch.cuda.is_available()
            or args.sampling_steps < 1 or not args.thresholds
            or any(not 0 < threshold < 1 for threshold in args.thresholds)
            or len(set(args.thresholds)) != len(args.thresholds)
            or args.ray_angle_deg <= 0 or args.ray_lateral_m <= 0
            or args.limit is not None and args.limit < 1):
        parser.error("CUDA, positive limits/steps, and unique thresholds in (0,1) are required")
    thresholds = tuple(sorted(args.thresholds))
    stage1, saved1 = load_stage1(args.stage1_checkpoint, args.device,
                                 args.radar_variant)
    stage2, saved2 = load_stage2(args.stage2_checkpoint, args.device)
    stage1_sha = sha256(args.stage1_checkpoint)
    if (saved2["stage1_sha256"] != stage1_sha
            or saved2["radar_variant"] != args.radar_variant
            or saved2.get("fault_policy_version") != FAULT_POLICY_VERSION
            or tuple(tuple(item) for item in saved2.get("fault_plan", ())) != FAULT_PLAN):
        raise ValueError("Stage-II checkpoint, Stage-I checkpoint, or severe fault policy differ")
    dataset = paired_dataset(args.vod_root, args.split, args.radar_variant,
                             include_clean=True, data_cache_root=args.data_cache_root)
    full_ids = [frame.frame_id for frame in dataset.frames]
    ids = full_ids[:args.limit] if args.limit else full_ids
    fault_seed = int(saved2["fault_seed"])
    fog_cache = SevereFogCache(args.fault_cache_root, args.split, full_ids,
                               seed=fault_seed)
    radius = float(saved1.get("radius_m", 3.2))
    gate = float(saved2["gate_threshold"])
    stage2_cache = None
    if args.stage2_cache_root:
        identity = cache_identity(args.stage1_checkpoint, args.vod_root,
                                  args.radar_variant, radius, gate,
                                  args.split, full_ids)
        stage2_cache = Stage2FixedCache(args.stage2_cache_root / args.split,
                                        identity)
        stage2_cache.check_complete()
    run_identity = {
        "version": VERSION, "source_sha256": _source_hash(),
        "stage1_sha256": stage1_sha,
        "stage2_sha256": sha256(args.stage2_checkpoint),
        "stage1_checkpoint": str(args.stage1_checkpoint.resolve()),
        "stage2_checkpoint": str(args.stage2_checkpoint.resolve()),
        "split": args.split, "frame_ids": ids,
        "radar_variant": args.radar_variant, "fault_type": args.fault_type,
        "fault_seed": fault_seed, "random_seed": args.random_seed,
        "stage1_radius_m": radius, "stage2_gate_threshold": gate,
        "stage2_roi_xyz_m": [*GRID.minimum_xyz, *GRID.maximum_xyz],
        "voxel_xyz_m": GRID.voxel_xyz, "tolerance_m": 0.2,
        "chamfer_preprocess_voxel_m": 0.125,
        "chamfer_definition": "mean(pred_to_clean_squared_L2)+mean(clean_to_pred_squared_L2)",
        "thresholds": thresholds,
        "sampling_steps": args.sampling_steps,
        "stage1_only": args.stage1_only,
        "ray_proxy": {"angle_deg": args.ray_angle_deg,
                      "lateral_m": args.ray_lateral_m,
                      "radial_margin_m": 0.5 * float(np.linalg.norm(GRID.voxel_xyz)),
                      "beam_ids_available": False},
    }
    args.output_root.mkdir(parents=True, exist_ok=True)
    frames_dir = args.output_root / "frames"
    frames_dir.mkdir(exist_ok=True)
    manifest_path = args.output_root / "run_config.json"
    if manifest_path.exists():
        if json.loads(manifest_path.read_text(encoding="utf-8")) != json.loads(json.dumps(run_identity)):
            raise ValueError(f"Existing sweep uses different settings: {manifest_path}")
    else:
        _atomic_json(manifest_path, run_identity)
    alpha_bars = cosine_alpha_bars(device=args.device)
    records = []
    for index in tqdm(range(len(ids)), desc="Stage I/II threshold sweep", unit="frame"):
        frame_id = ids[index]
        path = frames_dir / f"{frame_id}.json"
        if path.is_file():
            record = json.loads(path.read_text(encoding="utf-8"))
            if record["frame_id"] != frame_id:
                raise ValueError(f"Frame record mismatch: {path}")
        else:
            record = _frame_scores(
                dataset[index], frame_id, stage1=stage1, stage2=stage2,
                stage2_cache=stage2_cache, fog_cache=fog_cache,
                alpha_bars=alpha_bars, radius_m=radius,
                gate_threshold=gate, thresholds=thresholds,
                sampling_steps=args.sampling_steps,
                random_seed=args.random_seed, fault_seed=fault_seed,
                forced_fault=None if args.fault_type == "auto" else args.fault_type,
                ray_angle_deg=args.ray_angle_deg,
                ray_lateral_m=args.ray_lateral_m,
                device=args.device, stage1_only=args.stage1_only)
            _atomic_json(path, record)
        records.append(record)
    result = {"run_config": run_identity, "selection_metric": args.select_by,
              "stages": _summarize(
                  records, thresholds, args.select_by,
                  ("stage1", "stage1_active") if args.stage1_only
                  else ("stage1", "stage1_active", "stage2"))}
    if not args.stage1_only:
        result["stage2_false_positive_diagnostics"] = _summarize_false_positives(
            records, thresholds)
    _atomic_json(args.output_root / "threshold_metrics.json", result)
    fields = ("stage", "fault_type", "threshold", *METRICS,
              "predicted", "observed", "tp_exact", "tp_at_0_2m", "frames",
              *(f"faulty_{metric}" for metric in METRICS),
              "iou_exact_gain", "iou_at_0_2m_gain", "f1_exact_gain",
              "f1_at_0_2m_gain", "chamfer_squared_l2_reduction",
              "chamfer_paired_frames", "chamfer_faulty_paired_mean",
              "chamfer_model_paired_mean", "chamfer_defined_frames")
    with (args.output_root / "threshold_metrics.csv").open(
            "w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for stage, stage_result in result["stages"].items():
            for fault, scope in stage_result["scopes"].items():
                faulty = scope["faulty_baseline"]
                for row in scope["rows"]:
                    values = {"stage": stage, "fault_type": fault,
                              **row, **{f"faulty_{name}": faulty[name]
                                       for name in METRICS}}
                    writer.writerow({name: values.get(name) for name in fields})
    if not args.stage1_only:
        fp_fields = ("fault_type", "threshold", *FP_FIELDS,
                     "false_positive_fraction", "no_retained_return_fp_fraction",
                     "gate_allowed_cells", "gate_clean_cells", "gate_clean_cells_in_gate",
                     "gate_clean_cell_coverage", "gate_clean_return_fraction_of_allowed")
        with (args.output_root / "false_positive_diagnostics.csv").open(
                "w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fp_fields)
            writer.writeheader()
            for fault, scope in result["stage2_false_positive_diagnostics"].items():
                gate_fields = {f"gate_{key}": value for key, value in scope["gate"].items()}
                for row in scope["rows"]:
                    writer.writerow({**{"fault_type": fault}, **row, **gate_fields})
    for stage, stage_result in result["stages"].items():
        threshold = stage_result["selected_threshold"]
        selected = next(row for row in stage_result["scopes"]["overall"]["rows"]
                        if row["threshold"] == threshold)
        baseline = stage_result["scopes"]["overall"]["faulty_baseline"]
        print(f"{stage}: best {stage_result['selection_metric']} threshold={threshold:.3f}; "
              f"exact IoU {baseline['iou_exact']} -> {selected['iou_exact']}; "
              f"IoU@0.2m {baseline['iou_at_0_2m']} -> {selected['iou_at_0_2m']}; "
              f"F1 {baseline['f1_exact']} -> {selected['f1_exact']}; "
              f"F1@0.2m {baseline['f1_at_0_2m']} -> {selected['f1_at_0_2m']}; "
              f"Chamfer {baseline['chamfer_squared_l2']} -> "
              f"{selected['chamfer_squared_l2']} "
              f"(paired reduction {selected['chamfer_squared_l2_reduction']} "
              f"on {selected['chamfer_paired_frames']} frames)", flush=True)
    if not args.stage1_only:
        selected_stage2 = result["stages"]["stage2"]["selected_threshold"]
        diagnostics = result["stage2_false_positive_diagnostics"]["overall"]
        selected_diagnostic = next(row for row in diagnostics["rows"]
                                   if row["threshold"] == selected_stage2)
        print(f"Stage II at {selected_stage2:.3f}: "
              f"{selected_diagnostic['false_positive_additions']} exact-voxel FP additions; "
              f"{selected_diagnostic['known_free_before_clean_return']} before clean returns; "
              f"{selected_diagnostic['behind_retained_faulty_return']} behind retained returns; "
              f"{selected_diagnostic['no_retained_return_false_positives']} without a retained "
              f"ray match. Gate clean-cell coverage: "
              f"{diagnostics['gate']['clean_cell_coverage']}", flush=True)
    print(f"Saved metrics and false-positive diagnostics under {args.output_root}", flush=True)


if __name__ == "__main__":
    main()

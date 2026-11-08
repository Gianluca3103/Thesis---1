"""Compare lighter fog (severity 2 by default) against severity 5.

The preview applies the original simulator to raw VoD LiDAR, crops to the
Stage-II radar sector, and retains only original XYZ returns still present. It
does not alter an existing training cache.
"""

from __future__ import annotations

import argparse
from functools import lru_cache
import html
import importlib.util
import json
from pathlib import Path
import random

import numpy as np

from Fault_Localization_Model.vod_dataset.vod_io import (
    discover_vod_frames, load_vod_lidar, load_vod_radar_to_lidar,
)
from models.radar_lidar_stage1.radar_fov import crop_lidar_to_radar_fov
from models.radar_lidar_stage2.fault_data import retain_original_fog_returns
from models.radar_lidar_stage2.mild_fault_data import MILD_FAULT_VERSION
from scripts.visualize_stage2_mild_faults import (
    _find_mild_root, _removed_mask, _write_html, _write_ply,
)


FOG_ALPHA_BY_SEVERITY = {1: 0.005, 2: 0.01, 3: 0.02, 4: 0.03, 5: 0.06}


@lru_cache(maxsize=5)
def _fog_module(root: str, severity: int):
    source = Path(root) / "fog_simulation.py"
    if not source.is_file():
        raise FileNotFoundError(source)
    spec = importlib.util.spec_from_file_location(f"stage2_fog_preview_s{severity}", source)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load fog simulator: {source}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    alpha = FOG_ALPHA_BY_SEVERITY[severity]
    # The upstream simulator loads a large integral lookup on every call.
    # Keep the severity-specific table in this process for the selected frames.
    lookup = module.get_integral_dict(module.ParameterSet(alpha=alpha, gamma=0.000001))
    module.get_integral_dict = lambda _parameters: lookup
    return module


def _simulate_retained(raw: np.ndarray, clean_crop: np.ndarray,
                       lidar_from_radar: np.ndarray, fog_root: Path,
                       frame_id: str, severity: int, seed: int) -> np.ndarray:
    module = _fog_module(str(fog_root), severity)
    alpha = FOG_ALPHA_BY_SEVERITY[severity]
    injection_seed = int(np.random.SeedSequence(
        [seed, int(frame_id), severity]).generate_state(1)[0])
    module.RNG = np.random.default_rng(injection_seed)
    parameters = module.ParameterSet(alpha=alpha, gamma=0.000001)
    hard = module.P_R_fog_hard(parameters, raw.copy())
    simulated, _, _ = module.P_R_fog_soft(
        parameters, hard.copy(), raw[:, 3].copy(), noise=10,
        gain=False, noise_variant="v1")
    if simulated.shape != raw.shape or not np.isfinite(simulated).all():
        raise ValueError(f"Invalid fog simulation for {frame_id} severity {severity}")
    cropped = crop_lidar_to_radar_fov(
        np.asarray(simulated[:, :4], dtype=np.float32), lidar_from_radar)
    return retain_original_fog_returns(clean_crop, cropped)


def preview(vod_root: Path, fog_root: Path, output_root: Path, *, split: str,
            radar_variant: str, frame_ids: list[str], severity: int,
            seed: int, max_display_points: int) -> list[dict]:
    if severity not in (1, 2, 3, 4):
        raise ValueError("Lighter fog severity must be 1, 2, 3, or 4")
    frames = {frame.frame_id: frame for frame in discover_vod_frames(
        vod_root, split, radar_variant=radar_variant)}
    unknown = set(frame_ids) - set(frames)
    if unknown:
        raise ValueError(f"Frames absent from {split}: {sorted(unknown)}")
    output_root.mkdir(parents=True, exist_ok=True)
    rows = []
    for frame_id in frame_ids:
        frame = frames[frame_id]
        raw = np.asarray(load_vod_lidar(frame.lidar_path), dtype=np.float32)
        transform = load_vod_radar_to_lidar(
            frame.lidar_calibration_path, frame.radar_calibration_path)
        clean_crop = crop_lidar_to_radar_fov(raw, transform)
        clean_xyz = np.ascontiguousarray(clean_crop[:, :3], dtype=np.float32)
        pair = {}
        for level in (5, severity):
            faulty = _simulate_retained(raw, clean_crop, transform, fog_root,
                                        frame_id, level, seed)
            removed = clean_xyz[_removed_mask(clean_xyz, faulty)]
            if len(removed) != len(clean_xyz) - len(faulty):
                raise AssertionError(f"Fog point-count mismatch for {frame_id} severity {level}")
            stem = f"{split}_{frame_id}_fog_sim_{level}"
            row = {"split": split, "frame_id": frame_id,
                   "fault_type": f"fog_sim_{level}", "stem": stem,
                   "html": f"{stem}.html", "clean_points": len(clean_xyz),
                   "retained_points": len(faulty), "removed_points": len(removed),
                   "alpha": FOG_ALPHA_BY_SEVERITY[level]}
            for name, points in (("clean", clean_xyz), ("faulty", faulty),
                                 ("removed", removed)):
                _write_ply(output_root / f"{stem}_{name}.ply", points)
            _write_html(output_root / row["html"], row, clean_xyz, faulty,
                        removed, max_display_points, seed)
            pair[level] = row
        rows.append({"frame_id": frame_id, "strong": pair[5], "lighter": pair[severity]})
        print(f"{frame_id}: clean={len(clean_xyz):,}, severity 5 retained="
              f"{pair[5]['retained_points']:,} ({pair[5]['retained_points'] / max(1, len(clean_xyz)):.1%}), "
              f"severity {severity} retained={pair[severity]['retained_points']:,} "
              f"({pair[severity]['retained_points'] / max(1, len(clean_xyz)):.1%})",
              flush=True)
    lines = []
    for item in rows:
        strong, lighter = item["strong"], item["lighter"]
        lines.append(
            f'<tr><td>{html.escape(item["frame_id"])}</td>'
            f'<td><a href="{html.escape(strong["html"], quote=True)}">severity 5</a> · '
            f'{strong["retained_points"] / max(1, strong["clean_points"]):.1%} retained</td>'
            f'<td><a href="{html.escape(lighter["html"], quote=True)}">severity {severity}</a> · '
            f'{lighter["retained_points"] / max(1, lighter["clean_points"]):.1%} retained</td></tr>')
    page = ("<!doctype html><html lang=\"en\"><meta charset=\"utf-8\">"
            "<title>Stage II fog strength preview</title>"
            "<style>body{font:16px system-ui;max-width:900px;margin:35px auto;padding:0 18px;"
            "background:#101820;color:#eef5fa}a{color:#9dcafb}td,th{padding:10px;"
            "border-bottom:1px solid #3b5164;text-align:left}</style>"
            f"<h1>Stage II fog: severity 5 versus {severity}</h1>"
            "<p>Same validation frames and Stage II radar-sector crop. Each link opens "
    "an interactive clean/faulty/removed 3D view. Retention counts original "
    "clean XYZ returns still present regardless of intensity. This is a preview, not a training cache.</p>"
            "<table><tr><th>Frame</th><th>Current fog</th><th>Lighter fog</th></tr>"
            + "\n".join(lines) + "</table></html>")
    (output_root / "index.html").write_text(page, encoding="utf-8")
    (output_root / "summary.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"Open: {output_root / 'index.html'}", flush=True)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vod-root", type=Path,
                        help="VoD root; inferred from mild-cache manifest if omitted")
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--mild-root", type=Path)
    source.add_argument("--search-root", type=Path,
                        help="Find an existing mild-fault manifest to select fog frames")
    parser.add_argument("--fog-simulator-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "val"), default="val")
    parser.add_argument("--radar-variant", default="radar_5frames")
    parser.add_argument("--severity", type=int, choices=(1, 2, 3, 4), default=2)
    parser.add_argument("--count", type=int, default=3)
    parser.add_argument("--frame-id", action="append", dest="frame_ids")
    parser.add_argument("--seed", type=int, default=3103)
    parser.add_argument("--max-display-points", type=int, default=12000)
    args = parser.parse_args()
    if args.count < 1 or args.max_display_points < 1:
        parser.error("--count and --max-display-points must be positive")
    mild_root = args.mild_root or (_find_mild_root(args.search_root, args.split)
                                   if args.search_root is not None else None)
    manifest = None
    if mild_root is not None:
        manifest = json.loads((mild_root / args.split / "manifest.json").read_text(encoding="utf-8"))
        if manifest.get("version") != MILD_FAULT_VERSION:
            parser.error(f"Expected existing {MILD_FAULT_VERSION} cache: {mild_root}")
        print(f"Selecting fog frames from: {mild_root}", flush=True)
    vod_root = args.vod_root or (Path(manifest["vod_root"]) if manifest else None)
    if vod_root is None:
        parser.error("Pass --vod-root or an existing --mild-root/--search-root")
    radar_variant = manifest["radar_variant"] if manifest else args.radar_variant
    seed = manifest["seed"] if manifest else args.seed
    if args.frame_ids:
        frame_ids = args.frame_ids
    elif manifest:
        candidates = [frame_id for frame_id in manifest["frame_ids"]
                      if manifest["frame_faults"][frame_id] == "fog_sim_5"]
        frame_ids = random.Random(seed).sample(candidates, min(args.count, len(candidates)))
    else:
        frames = discover_vod_frames(vod_root, args.split, radar_variant=radar_variant)
        frame_ids = random.Random(seed).sample(
            [frame.frame_id for frame in frames], min(args.count, len(frames)))
    if not frame_ids:
        parser.error("No frames available for the preview")
    preview(vod_root, args.fog_simulator_root, args.output_root,
            split=args.split, radar_variant=radar_variant, frame_ids=frame_ids,
            severity=args.severity, seed=seed,
            max_display_points=args.max_display_points)


if __name__ == "__main__":
    main()

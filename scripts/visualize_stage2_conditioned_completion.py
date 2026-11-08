"""Show conditioned clean-latent denoiser additions against faulty and clean LiDAR.

This traverses the full split to reproduce the fixed-noise sequence used by
the threshold evaluator, then saves a balanced set of lightweight examples.
The clean latent is an oracle input to this diagnostic model.
"""

from __future__ import annotations

import argparse
import html
import json
import random
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.radar_lidar_crossmodal_v1.pipeline_data import paired_dataset
from models.radar_lidar_crossmodal_v1.radar_pruner_cache import (
    RadarSupportCache, identity as support_identity)
from models.radar_lidar_stage1.sparse import encode_keys
from models.radar_lidar_stage2.clean_selected_metrics import (
    add_frame, completion_summary, empty_accumulator, full_frame_voxels,
    voxel_centers)
from models.radar_lidar_stage2.geometry_gate import offset_targets
from models.radar_lidar_stage2.sparse_geometry import GRID
from models.radar_lidar_stage2.sparse_residual import (
    LPCR_REVISION, VERSION, SparseResidualStage2, cosine_alpha_bars)
from models.radar_lidar_stage2.stage2_io import sha256
from scripts.train_stage2_clean_denoiser import (
    CONDITIONED_ONE_STEP_REVISION, MIN_RADAR_VOXELS, RADIUS_M,
    _batch, _conditioned_inputs, _CleanRows, _rows, sample_clean_latent)
from scripts.visualize_stage2_lpcr import _write_ply


def balanced_frame_ids(frame_ids: list[str], faults: dict[str, str],
                       per_fault: int, seed: int) -> list[str]:
    """Choose the same number of deterministic examples for each fault."""
    if per_fault < 1 or set(frame_ids) != set(faults):
        raise ValueError("Need positive per-fault count and complete assignments")
    randomizer = random.Random(seed)
    groups: dict[str, list[str]] = {}
    for frame_id in frame_ids:
        groups.setdefault(faults[frame_id], []).append(frame_id)
    selected = set()
    for fault in sorted(groups):
        if len(groups[fault]) < per_fault:
            raise ValueError(f"Only {len(groups[fault])} {fault} frames available")
        selected.update(randomizer.sample(groups[fault], per_fault))
    return [frame_id for frame_id in frame_ids if frame_id in selected]


def _display(xyz: np.ndarray, limit: int, seed: int) -> list[list[float]]:
    if len(xyz) > limit:
        indices = np.sort(np.random.default_rng(seed).choice(
            len(xyz), limit, replace=False))
        xyz = xyz[indices]
    return np.round(xyz.astype(np.float64), 3).tolist()


def _roi_radar(points: torch.Tensor) -> np.ndarray:
    xyz = points[:, :3].cpu().numpy()
    lower = np.asarray(GRID.minimum_xyz)
    upper = np.asarray(GRID.maximum_xyz)
    return xyz[((xyz >= lower) & (xyz < upper)).all(1)]


def _plot(path: Path, record: dict) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11, 8.5), dpi=110,
                             constrained_layout=True)
    fig.patch.set_facecolor("#0c1620")
    for axis in axes.flat:
        axis.set_facecolor("#0c1620")
        axis.set_xlim(0, 32)
        axis.set_ylim(-16, 16)
        axis.set_aspect("equal")
        axis.tick_params(colors="#cbd7df", labelsize=8)
        axis.set_xlabel("X (m)", color="#cbd7df")
        axis.set_ylabel("Y (m)", color="#cbd7df")
    plots = (
        ("Clean target", ((record["clean_xyz"], "#dbe7ef", 1.0),)),
        ("Faulty input", ((record["faulty_xyz"], "#68b8ff", 1.0),)),
        ("New cells after threshold", (
            (record["faulty_xyz"], "#526b81", 0.7),
            (record["true_added_xyz"], "#48e1ad", 2.0),
            (record["false_added_xyz"], "#ff9c57", 2.0))),
        ("Completed scan and misses", (
            (record["faulty_xyz"], "#68b8ff", 0.8),
            (record["true_added_xyz"], "#48e1ad", 1.8),
            (record["false_added_xyz"], "#ff9c57", 1.8),
            (record["missing_xyz"], "#ff5964", 1.8))))
    for axis, (title, layers) in zip(axes.flat, plots):
        axis.set_title(title, color="white")
        for xyz, color, size in layers:
            if len(xyz):
                axis.scatter(xyz[:, 0], xyz[:, 1], s=size, c=color,
                             linewidths=0)
    before = record["overall"]["faulty"]
    after = record["overall"]["completed"]
    source_note = (
        "Oracle-assisted diagnostic: denoising starts from clean-LiDAR latent"
        if record.get("oracle_clean_latent", True) else
        "Prediction starts from Gaussian noise; inputs are faulty LiDAR and radar")
    fig.suptitle(
        f"{record['split']} {record['frame_id']} · {record['fault']} · "
        f"epoch {record['epoch']} · threshold {record['threshold']:.3f}\n"
        f"Full-ROI IoU {before['iou']:.1%} → {after['iou']:.1%}; "
        f"green: correct addition, orange: false addition, red: still missing\n"
        f"{source_note}",
        color="white", fontsize=11)
    fig.savefig(path, facecolor=fig.get_facecolor())
    plt.close(fig)


def _viewer(path: Path, record: dict, seed: int) -> None:
    layers = []
    for index, (name, color, xyz, limit) in enumerate((
            ("Clean target", "#dbe7ef", record["clean_xyz"], 6000),
            ("Faulty input", "#68b8ff", record["faulty_xyz"], 6000),
            ("New true additions", "#48e1ad", record["true_added_xyz"], 5000),
            ("New false additions", "#ff9c57", record["false_added_xyz"], 5000),
            ("Still missing", "#ff5964", record["missing_xyz"], 4000),
            ("Radar returns", "#7bbdff", record["radar_xyz"], 2000),
            ("Eligible gate (subset)", "#ae89df", record["gate_xyz"], 3000))):
        layers.append({"name": name, "color": color, "full": len(xyz),
                       "points": _display(xyz, limit, seed + index)})
    metrics = record["overall"]
    payload = json.dumps({
        "split": record["split"], "frame": record["frame_id"],
        "fault": record["fault"], "epoch": record["epoch"],
        "threshold": record["threshold"], "layers": layers,
        "before_iou": metrics["faulty"]["iou"],
        "after_iou": metrics["completed"]["iou"],
        "before_iou_0p2m": metrics["faulty"]["iou_0p2m"],
        "after_iou_0p2m": metrics["completed"]["iou_0p2m"],
        "added_cells": metrics["added_cells"]},
        separators=(",", ":")).replace("<", "\\u003c")
    markup = r'''<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Stage II conditioned completion diagnostic</title>
<style>body{margin:0;padding:12px;background:#101820;color:#edf4fa;font:14px system-ui}
a{color:#9dcafb}button{background:#253847;color:inherit;border:1px solid #637b8b;border-radius:4px;padding:5px 9px;cursor:pointer}
.controls,.layers{display:flex;gap:8px 14px;flex-wrap:wrap;align-items:center;margin:9px 0}
label{white-space:nowrap}canvas{width:100%;height:72vh;min-height:440px;background:#0c1620;touch-action:none;cursor:grab}</style>
<a href="index.html">← All samples</a><h2 id="title"></h2><p id="note"></p>
<div class="controls"><button id="clean">Clean</button><button id="faulty">Faulty</button>
<button id="completed">Completed</button><button id="errors">Errors</button>
<button id="gate">Radar gate</button><button id="zin">Zoom in</button>
<button id="zout">Zoom out</button><button id="reset">Reset</button></div>
<div id="layers" class="layers"></div><canvas id="view"></canvas>
<p>Drag to rotate · Shift-drag to pan · wheel to zoom. Points are occupancy voxel centers. Display is sampled; PLY and NPZ contain every voxel.</p>
<p><b>Interpretation:</b> __INTERPRETATION__</p>
<p id="downloads"></p><script id="data" type="application/json">__DATA__</script>
<script>(()=>{const d=JSON.parse(document.getElementById('data').textContent);
document.getElementById('title').textContent=`${d.split} ${d.frame} · ${d.fault} · epoch ${d.epoch} · threshold ${d.threshold.toFixed(3)}`;
document.getElementById('note').textContent=`Full-ROI exact IoU ${Math.round(d.before_iou*1000)/10}% → ${Math.round(d.after_iou*1000)/10}%; IoU at 0.2 m ${Math.round(d.before_iou_0p2m*1000)/10}% → ${Math.round(d.after_iou_0p2m*1000)/10}%; ${d.added_cells.toLocaleString()} new cells.`;
const stem=`${d.split}_${d.frame}`;document.getElementById('downloads').innerHTML=`<a href="${stem}.png">PNG</a> · <a href="${stem}.npz">Full NPZ</a> · <a href="${stem}_faulty.ply">Faulty PLY</a> · <a href="${stem}_completed.ply">Completed PLY</a> · <a href="${stem}_clean.ply">Clean PLY</a>`;
const panel=document.getElementById('layers'),checks=[];
for(let i=0;i<d.layers.length;i++){const l=d.layers[i],label=document.createElement('label'),input=document.createElement('input');input.type='checkbox';checks.push(input);label.append(input,document.createTextNode(` ${l.name} (${l.full.toLocaleString()} full; ${l.points.length.toLocaleString()} shown)`));label.style.color=l.color;panel.append(label);input.addEventListener('change',draw);}
const canvas=document.getElementById('view'),view={yaw:-.65,pitch:.35,zoom:1,panX:0,panY:0};
function project(p,w,h,s){const x=p[0]-16,y=p[1],z=p[2],cy=Math.cos(view.yaw),sy=Math.sin(view.yaw),cp=Math.cos(view.pitch),sp=Math.sin(view.pitch),u=cy*x-sy*y,v=sy*x+cy*y;return [w/2+view.panX+u*s,h/2+view.panY-(cp*z-sp*v)*s];}
let pending=false;function draw(){if(pending)return;pending=true;requestAnimationFrame(()=>{pending=false;const ratio=devicePixelRatio||1,w=canvas.clientWidth,h=canvas.clientHeight;if(!w||!h)return;canvas.width=Math.round(w*ratio);canvas.height=Math.round(h*ratio);const ctx=canvas.getContext('2d');ctx.setTransform(ratio,0,0,ratio,0,0);ctx.clearRect(0,0,w,h);const s=.90*Math.min(w,h)*view.zoom/46;for(let i=0;i<d.layers.length;i++){if(!checks[i].checked)continue;const l=d.layers[i];ctx.fillStyle=l.color;for(const point of l.points){const [x,y]=project(point,w,h,s);if(x>=0&&x<w&&y>=0&&y<h)ctx.fillRect(x-1,y-1,2,2);}}});}
function preset(on){checks.forEach((item,i)=>item.checked=on.includes(i));draw();}
document.getElementById('clean').onclick=()=>preset([0]);document.getElementById('faulty').onclick=()=>preset([1]);document.getElementById('completed').onclick=()=>preset([1,2,3]);document.getElementById('errors').onclick=()=>preset([2,3,4]);document.getElementById('gate').onclick=()=>preset([0,5,6]);
function zoom(k){view.zoom=Math.max(.1,Math.min(80,view.zoom*k));draw();}let last=null;
canvas.addEventListener('pointerdown',e=>{canvas.setPointerCapture(e.pointerId);last=[e.clientX,e.clientY];});
canvas.addEventListener('pointermove',e=>{if(!last)return;const dx=e.clientX-last[0],dy=e.clientY-last[1];if(e.shiftKey){view.panX+=dx;view.panY+=dy;}else{view.yaw+=dx*.008;view.pitch=Math.max(-1.55,Math.min(1.55,view.pitch+dy*.008));}draw();});
for(const event of ['pointerup','pointercancel'])canvas.addEventListener(event,()=>last=null);
canvas.addEventListener('wheel',e=>{e.preventDefault();zoom(Math.exp(-e.deltaY*.001));},{passive:false});
document.getElementById('zin').onclick=()=>zoom(1.6);document.getElementById('zout').onclick=()=>zoom(1/1.6);document.getElementById('reset').onclick=()=>{Object.assign(view,{yaw:-.65,pitch:.35,zoom:1,panX:0,panY:0});draw();};window.addEventListener('resize',draw);preset([1,2,3]);})();</script></html>'''
    interpretation = ("This model begins from a noised clean-LiDAR latent. "
                      "These are oracle-assisted diagnostic outputs."
                      if record.get("oracle_clean_latent", True) else
                      "Prediction begins from Gaussian noise with faulty "
                      "LiDAR and radar conditioning. Clean LiDAR is loaded "
                      "only afterward to score and color the result.")
    path.write_text(markup.replace("__DATA__", payload).replace(
        "__INTERPRETATION__", interpretation), encoding="utf-8")


def _save_record(root: Path, record: dict, seed: int) -> dict:
    stem = f"{record['split']}_{record['frame_id']}"
    np.savez_compressed(
        root / f"{stem}.npz", clean_bzyx=record["clean"].numpy(),
        faulty_bzyx=record["faulty"].numpy(),
        completed_bzyx=record["completed"].numpy(),
        added_bzyx=record["added"].numpy(),
        gate_bzyx=record["gate"].numpy(),
        gate_probability=record["probability"],
        radar_xyz=record["radar_xyz"],
        threshold=record["threshold"], checkpoint_epoch=record["epoch"])
    for name in ("clean", "faulty", "completed", "added"):
        _write_ply(root / f"{stem}_{name}.ply",
                   record[f"{name}_xyz"].astype(np.float32))
    _plot(root / f"{stem}.png", record)
    _viewer(root / f"{stem}.html", record, seed)
    return {"frame_id": record["frame_id"], "fault": record["fault"],
            "added_cells": record["overall"]["added_cells"],
            "overall": record["overall"], "html": f"{stem}.html",
            "png": f"{stem}.png"}


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vod-root", type=Path, required=True)
    parser.add_argument("--data-cache-root", type=Path, required=True)
    parser.add_argument("--support-cache-root", type=Path, required=True)
    parser.add_argument("--faulty-root", type=Path, required=True)
    parser.add_argument("--lpcr-checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--sweep-json", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument("--per-fault", type=int, default=2)
    parser.add_argument("--seed", type=int, default=3103)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.device != "cuda" or not torch.cuda.is_available():
        parser.error("CUDA is required")
    saved = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    identity = saved.get("identity")
    sweep = json.loads(args.sweep_json.read_text(encoding="utf-8"))
    if (not isinstance(identity, dict) or
            identity.get("revision") != CONDITIONED_ONE_STEP_REVISION or
            identity.get("lpcr_sha256") != sha256(args.lpcr_checkpoint) or
            sweep.get("checkpoint_sha256") != sha256(args.checkpoint) or
            sweep.get("checkpoint_epoch") != saved.get("epoch") or
            sweep.get("model_revision") != identity["revision"] or
            sweep.get("conditioning") != "faulty_LiDAR_and_measured_radar" or
            sweep.get("sampling") != "one_step_clean_x0_at_fixed_t" or
            sweep.get("eval_timestep") != identity["eval_timestep"]):
        raise ValueError("Checkpoint, LPCR and conditioned threshold sweep differ")
    threshold = float(sweep["selected"]["best_iou"]["threshold"])
    step = int(sweep["threshold_steps"])
    if step != 1000 or not 0 <= threshold <= 1:
        raise ValueError("Unexpected threshold resolution")
    step_index = round(threshold * step)
    for split in ("train", "val"):
        for label, root, key in (
                ("fixed", args.data_cache_root, "data_manifest_sha256"),
                ("radar support", args.support_cache_root,
                 "support_manifest_sha256"),
                ("mild faults", args.faulty_root, "faulty_manifest_sha256")):
            if identity[key][split] != sha256(root / split / "manifest.json"):
                raise ValueError(f"{label} {split} cache differs from training")
    dataset = paired_dataset(args.vod_root, args.split, identity["radar_variant"],
                             include_clean=True,
                             data_cache_root=args.data_cache_root)
    frame_ids = [frame.frame_id for frame in dataset.frames]
    manifest = json.loads((args.faulty_root / args.split / "manifest.json")
                          .read_text(encoding="utf-8"))
    if manifest.get("frame_ids") != frame_ids or manifest.get("split") != args.split:
        raise ValueError("Fault-cache frame IDs do not match this split")
    if args.split == "test":
        training = json.loads((args.faulty_root / "train" / "manifest.json")
                              .read_text(encoding="utf-8"))
        for key in ("version", "seed", "radar_variant", "fault_types",
                    "assignment_policy", "partial_loss_policy"):
            if manifest.get(key) != training.get(key):
                raise ValueError(f"Test mild-fault policy differs: {key}")
    selected_ids = set(balanced_frame_ids(
        frame_ids, manifest["frame_faults"], args.per_fault, args.seed))
    cache = RadarSupportCache(
        args.support_cache_root / args.split,
        support_identity(args.split, frame_ids, identity["radar_variant"],
                         data_cache_root=args.data_cache_root,
                         radius_m=RADIUS_M,
                         min_radar_voxels=MIN_RADAR_VOXELS))
    cache.check_complete()
    loader = DataLoader(
        _CleanRows(dataset, cache, None, args.faulty_root, args.split,
                   include_full_clean=True),
        batch_size=identity["batch_size"], shuffle=False,
        num_workers=args.num_workers, collate_fn=_rows, pin_memory=True,
        persistent_workers=args.num_workers > 0)
    saved_lpcr = torch.load(args.lpcr_checkpoint, map_location="cpu",
                            weights_only=False)
    if (saved_lpcr.get("version") != VERSION or
            saved_lpcr.get("model_revision") != LPCR_REVISION or
            saved_lpcr.get("phase") != "lpcr" or
            not saved_lpcr.get("lpcr_full_region")):
        raise ValueError("Expected frozen full-region LPCR checkpoint")
    model = SparseResidualStage2(diffusion_target="epsilon").to(args.device)
    missing, unexpected = model.load_state_dict(saved_lpcr["model"], strict=False)
    if unexpected or any(not key.startswith("addition_head.") for key in missing):
        raise ValueError("LPCR checkpoint differs from architecture")
    model.diffusion.load_state_dict(saved["diffusion"])
    model.radar_adapter.load_state_dict(saved["radar_adapter"])
    model.eval().requires_grad_(False)
    alpha_bars = cosine_alpha_bars(device=args.device)
    generator = torch.Generator(device=args.device)
    generator.manual_seed(int(identity["seed"]))
    records = []
    for rows in tqdm(loader, desc=f"Conditioned completion {args.split}",
                     unit="batch"):
        hierarchy, clean, point_ids = _batch(rows, args.device)
        fine, coarse = hierarchy[0], hierarchy[-1]
        if len(coarse):
            x0, _ = model.lidar_encoder(clean, coarse, point_ids)
            conditions = _conditioned_inputs(model, rows, hierarchy, args.device)
            recovered = sample_clean_latent(
                model, x0, coarse, alpha_bars, generator,
                batch_size=len(rows), strategy="one_step",
                eval_timestep=int(identity["eval_timestep"]),
                conditions=conditions)
            logits, _ = model.lpcr(recovered.clamp(-1, 1), hierarchy,
                                   x0.new_ones((len(fine), 1)))
            selected = (torch.floor(logits.sigmoid() * step).long()
                        .clamp(0, step) >= step_index)
        else:
            selected = torch.zeros(len(fine), dtype=torch.bool,
                                   device=args.device)
        for index, row in enumerate(rows):
            frame_id = row["frame_id"]
            if frame_id not in selected_ids:
                continue
            frame_mask = fine[:, 0] == index
            gate = fine[frame_mask].cpu().clone()
            gate[:, 0] = 0
            predicted = fine[frame_mask & selected].cpu().clone()
            predicted[:, 0] = 0
            probabilities = (logits.sigmoid()[frame_mask].cpu().numpy()
                             if len(coarse) else np.empty(0, dtype=np.float32))
            clean_full, faulty, completed = full_frame_voxels(
                row["full_clean_xyz"], row["faulty_xyz"], predicted)
            clean_keys = encode_keys(clean_full, GRID.shape_zyx)
            faulty_keys = encode_keys(faulty, GRID.shape_zyx)
            completed_keys = encode_keys(completed, GRID.shape_zyx)
            added = completed[~torch.isin(completed_keys, faulty_keys)]
            added_keys = encode_keys(added, GRID.shape_zyx)
            true_add = added[torch.isin(added_keys, clean_keys)]
            false_add = added[~torch.isin(added_keys, clean_keys)]
            missing_clean = clean_full[~torch.isin(clean_keys, completed_keys)]
            before, after = empty_accumulator(), empty_accumulator()
            add_frame(before, faulty, clean_full)
            add_frame(after, completed, clean_full)
            overall = completion_summary(before, after)
            records.append({
                "frame_id": frame_id, "split": args.split,
                "fault": manifest["frame_faults"][frame_id],
                "epoch": int(saved["epoch"]), "threshold": threshold,
                "clean": clean_full, "faulty": faulty,
                "completed": completed, "added": added, "gate": gate,
                "probability": probabilities, "overall": overall,
                "clean_xyz": voxel_centers(clean_full),
                "faulty_xyz": voxel_centers(faulty),
                "completed_xyz": voxel_centers(completed),
                "added_xyz": voxel_centers(added),
                "true_added_xyz": voxel_centers(true_add),
                "false_added_xyz": voxel_centers(false_add),
                "missing_xyz": voxel_centers(missing_clean),
                "gate_xyz": voxel_centers(gate),
                "radar_xyz": _roi_radar(row["radar"])})
    if len(records) != len(selected_ids):
        raise RuntimeError(f"Captured {len(records)} of {len(selected_ids)} frames")
    args.output_root.mkdir(parents=True, exist_ok=True)
    summaries = []
    for index, record in enumerate(records):
        summary = _save_record(args.output_root, record, args.seed + index * 11)
        summaries.append(summary)
        before = summary["overall"]["faulty"]
        after = summary["overall"]["completed"]
        print(f"{record['frame_id']} {record['fault']}: "
              f"full-ROI IoU {before['iou']:.2%} -> {after['iou']:.2%}; "
              f"{summary['added_cells']:,} added", flush=True)
    (args.output_root / "summary.json").write_text(
        json.dumps({"checkpoint": str(args.checkpoint),
                    "checkpoint_sha256": sha256(args.checkpoint),
                    "epoch": int(saved["epoch"]), "split": args.split,
                    "threshold": threshold, "oracle_clean_latent": True,
                    "samples": summaries}, indent=2) + "\n", encoding="utf-8")
    cards = "\n".join(
        f'<li><a href="{html.escape(item["html"], quote=True)}">'
        f'{html.escape(item["frame_id"])} · {html.escape(item["fault"])}'
        f'</a> · <a href="{html.escape(item["png"], quote=True)}">PNG</a></li>'
        for item in summaries)
    (args.output_root / "index.html").write_text(
        "<!doctype html><html lang='en'><meta charset='utf-8'>"
        "<title>Conditioned completion samples</title>"
        "<style>body{margin:30px auto;max-width:850px;padding:0 18px;"
        "background:#101820;color:#eef5fa;font:16px system-ui}"
        "a{color:#9dcafb}li{margin:13px 0}</style>"
        f"<h1>Conditioned completion diagnostic · {args.split}</h1>"
        "<p>Faulty scan, newly added occupancy voxels, combined scan, and "
        "clean target, all shown as voxel centers. "
        "Each page is a lightweight interactive 3D viewer with full-resolution "
        "NPZ and PLY downloads. The model begins from a noised clean-LiDAR "
        "latent, so this is an oracle-assisted diagnostic.</p><ul>"
        + cards + "</ul></html>", encoding="utf-8")
    print(f"3D viewer: {args.output_root / 'index.html'}", flush=True)
    print(f"PNG examples: {args.output_root}", flush=True)


if __name__ == "__main__":
    main()

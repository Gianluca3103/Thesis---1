"""Reconstruct Stage-I-allowed LiDAR sites with radar/faulty-conditioned Stage II."""

from __future__ import annotations

import argparse
import hashlib
import html
import json
from pathlib import Path
import random

import numpy as np
import torch

from models.radar_lidar_crossmodal_v1.pipeline_data import paired_dataset
from models.radar_lidar_stage1.sparse import encode_keys
from models.radar_lidar_stage2.fault_data import (
    FAULT_NAMES, FAULT_PLAN, FAULT_POLICY_VERSION, SevereFogCache,
    load_faulty, severe_fault,
)
from models.radar_lidar_stage2.sparse_geometry import GRID, reconstruction_targets
from models.radar_lidar_stage2.sparse_residual import cosine_alpha_bars
from models.radar_lidar_stage2.stage2_cache import Stage2FixedCache, cache_identity
from models.radar_lidar_stage2.stage2_io import (
    load_stage1, load_stage2, sha256, stage1_frame,
)


def _write_ply(path: Path, xyz: np.ndarray, probabilities: np.ndarray) -> None:
    colors = np.clip(probabilities * 255, 0, 255).astype(np.uint8)
    with path.open("w", encoding="utf-8") as stream:
        stream.write("ply\nformat ascii 1.0\n")
        stream.write(f"element vertex {len(xyz)}\n")
        for name in ("x", "y", "z"):
            stream.write(f"property float {name}\n")
        for name in ("red", "green", "blue"):
            stream.write(f"property uchar {name}\n")
        stream.write("end_header\n")
        for point, color in zip(xyz, colors):
            stream.write(f"{point[0]:.6f} {point[1]:.6f} {point[2]:.6f} "
                         f"{int(color)} {int(color)} {255-int(color)}\n")


def _roi(points: np.ndarray) -> np.ndarray:
    minimum = np.asarray(GRID.minimum_xyz, dtype=np.float32)
    maximum = np.asarray(GRID.maximum_xyz, dtype=np.float32)
    xyz = points[:, :3].astype(np.float32, copy=False)
    return points[((xyz >= minimum) & (xyz < maximum)).all(axis=1)]


def _present_in_faulty(clean: np.ndarray, faulty: np.ndarray) -> np.ndarray:
    """Match exact point rows, including reflectivity when present."""
    if not len(clean):
        return np.empty(0, dtype=bool)
    if clean.ndim != 2 or faulty.ndim != 2 or clean.shape[1] != faulty.shape[1]:
        raise ValueError("Point clouds must have the same feature width")
    clean_values = np.ascontiguousarray(clean, dtype=np.float32)
    faulty_values = np.ascontiguousarray(faulty, dtype=np.float32)
    row_dtype = np.dtype((np.void, clean.shape[1] * np.dtype(np.float32).itemsize))
    return np.isin(clean_values.view(row_dtype).reshape(-1),
                   faulty_values.view(row_dtype).reshape(-1))


def _write_html(path: Path, *, frame_id: str, epoch: int, split: str,
                faulty: np.ndarray, clean: np.ndarray, predicted: np.ndarray,
                removed: np.ndarray,
                threshold: float, allowed_cells: int, predicted_cells: int,
                sampling_steps: int,
                fault_type: str, totals: tuple[int, int, int],
                changed_geometry: int | None,
                changed_reflectivity: int | None,
                median_intensity: tuple[float | None, float | None],
                shared_subset: bool,
                metrics: dict | None) -> None:
    clouds = [faulty, clean, predicted]
    nonempty = [cloud for cloud in clouds if len(cloud)]
    if nonempty:
        combined = np.concatenate(nonempty)
        minimum, maximum = combined.min(0).tolist(), combined.max(0).tolist()
    else:
        minimum, maximum = list(GRID.minimum_xyz), list(GRID.maximum_xyz)
    payload = json.dumps({
        "frame": frame_id, "epoch": epoch, "split": split,
        "faulty": np.round(faulty, 3).tolist(),
        "clean": np.round(clean, 3).tolist(),
        "predicted": np.round(predicted, 3).tolist(),
        "removed": np.round(removed, 3).tolist(),
        "threshold": threshold, "allowed_cells": allowed_cells,
        "predicted_cells": predicted_cells,
        "sampling_steps": sampling_steps, "fault_type": fault_type,
        "totals": totals, "changed_geometry": changed_geometry,
        "changed_reflectivity": changed_reflectivity,
        "median_intensity": median_intensity,
        "shared_subset": shared_subset,
        "metrics": metrics,
        "minimum": minimum, "maximum": maximum,
    }, separators=(",", ":")).replace("<", "\\u003c")
    markup = r'''<!doctype html>
<html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Stage II diffusion reconstruction</title>
<style>
body{margin:0;padding:12px;background:#101820;color:#eef5fa;font:14px system-ui,sans-serif}
h1{font-size:20px;margin:0 0 6px}p{color:#b9c8d6;margin:5px 0 10px}
.controls{display:flex;gap:10px;margin:12px 0}button{background:#253847;color:inherit;border:1px solid #637b8b;border-radius:4px;padding:5px 9px;cursor:pointer}
.panels{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:8px}
section{min-width:0;background:#182633;border:1px solid #3b5164;border-radius:5px;padding:6px}
h2{font-size:14px;margin:0 0 5px}canvas{width:100%;height:48vh;min-height:340px;background:#0c1620;touch-action:none;cursor:grab}
.legend{color:#b9c8d6;margin-top:8px}
@media(max-width:750px){.panels{display:block}section{margin-bottom:8px}canvas{height:50vh;min-height:280px}}
</style>
<h1 id="title"></h1><p id="note"></p><p id="counts"></p>
<div class="controls"><button id="zin">Zoom in</button><button id="zout">Zoom out</button><button id="reset">Reset view</button></div>
<div id="panels" class="panels"></div>
<p class="legend"><span style="color:#ffb85c">●</span> faulty input &nbsp; <span style="color:#60d796">●</span> clean reference &nbsp; <span style="color:#f450d5">●</span> clean points changed in XYZ or reflectivity &nbsp; <span style="color:#ff8b6c">●</span> reconstruction. Drag to rotate; Shift-drag to pan; wheel to zoom. All panels share one camera. PLY and NPZ contain the complete reconstruction.</p>
<script id="data" type="application/json">__DATA__</script>
<script>
(() => {
const d=JSON.parse(document.getElementById('data').textContent);
document.getElementById('title').textContent=`${d.split} frame ${d.frame} · ${d.fault_type} · diffusion epoch ${d.epoch}`;
const explanations={total_loss:'All LiDAR returns are removed.',fov_filter:'The established FOV filter at severity 5 retains only its narrowest angular sector.',fog_sim:'The established fog simulator at severity 5 changes ranges and intensities on the full raw scan before the forward crop.',real:'Pre-aligned real faulty LiDAR was loaded.'};
document.getElementById('note').textContent=`${explanations[d.fault_type]} The model receives faulty LiDAR and radar features; clean LiDAR is used only as the reference.`;
document.getElementById('counts').textContent=`${d.totals[0].toLocaleString()} faulty points, ${d.totals[1].toLocaleString()} clean reference points, ${d.totals[2].toLocaleString()} reconstructed points in ${d.predicted_cells.toLocaleString()} predicted occupied voxels. ${d.changed_geometry===null?'Clean-point difference unavailable for real faulty LiDAR':`${d.changed_geometry.toLocaleString()} clean points absent or shifted in XYZ; ${d.changed_reflectivity.toLocaleString()} additional points changed only in reflectivity.`} Median reflectivity: clean ${d.median_intensity[0]===null?'unavailable':d.median_intensity[0].toFixed(1)}, faulty ${d.median_intensity[1]===null?'unavailable':d.median_intensity[1].toFixed(1)}. ${d.allowed_cells.toLocaleString()} radar-allowed cells; occupancy threshold ${d.threshold.toFixed(2)}; ${d.sampling_steps} diffusion steps. ${d.shared_subset?'Faulty and clean panels show the complete clean-point set.':'Fog and clean panels show their complete respective clouds, including shifted fog returns.'}`;
if(d.metrics){const p=document.createElement('p');p.textContent=`Reconstruction: Known IoU ${d.metrics.known_iou.toFixed(4)}, Full IoU ${d.metrics.full_iou.toFixed(4)}, gate coverage ${d.metrics.gate_coverage.toFixed(4)}. Full IoU counts predictions without a measured clean return as errors.`;document.getElementById('counts').after(p);}
const names=['Faulty LiDAR input',d.changed_geometry===null?'Clean LiDAR reference':'Clean LiDAR reference + changed points','Stage II reconstruction','Clean + reconstruction'];if(d.changed_geometry!==null)names.push('Faulty input + changed clean points');const panes=document.getElementById('panels'),canvases=[],headings=[];
for(const name of names){const section=document.createElement('section'),h=document.createElement('h2'),canvas=document.createElement('canvas');h.textContent=name;section.append(h,canvas);panes.append(section);canvases.push(canvas);headings.push(h);}
const mid=d.minimum.map((x,i)=>(x+d.maximum[i])/2),extent=Math.hypot(...d.minimum.map((x,i)=>d.maximum[i]-x))||1;
const initial={yaw:-.65,pitch:.35,zoom:1,panX:0,panY:0},view={...initial};
function project(p,w,h,s){const x=p[0]-mid[0],y=p[1]-mid[1],z=p[2]-mid[2],cy=Math.cos(view.yaw),sy=Math.sin(view.yaw),cp=Math.cos(view.pitch),sp=Math.sin(view.pitch);const u=cy*x-sy*y,v=sy*x+cy*y;return [w/2+view.panX+u*s,h/2+view.panY-(cp*z-sp*v)*s];}
function plot(ctx,points,w,h,s,color){ctx.fillStyle=color;for(const p of points){const [x,y]=project(p,w,h,s);if(x>=0&&x<w&&y>=0&&y<h)ctx.fillRect(x-1,y-1,2,2);}}
function draw(canvas,index){const ratio=devicePixelRatio||1,w=canvas.clientWidth,h=canvas.clientHeight;if(!w||!h)return;canvas.width=Math.round(w*ratio);canvas.height=Math.round(h*ratio);const ctx=canvas.getContext('2d');ctx.setTransform(ratio,0,0,ratio,0,0);ctx.clearRect(0,0,w,h);const scale=.88*Math.min(w,h)*view.zoom/extent;if(index===0||index===4)plot(ctx,d.faulty,w,h,scale,'#ffb85c');if(index===1||index===3)plot(ctx,d.clean,w,h,scale,'#60d796');if(index===2||index===3)plot(ctx,d.predicted,w,h,scale,'#ff8b6c');if(index===1||index===4)plot(ctx,d.removed,w,h,scale,'#f450d5');const count=index===0?d.faulty.length:index===1?d.clean.length:index===2?d.predicted.length:index===3?d.clean.length+d.predicted.length:d.faulty.length+d.removed.length;headings[index].textContent=`${names[index]}: ${count.toLocaleString()} shown`;const origin=project([0,0,0],w,h,scale);for(const [name,tint,end] of [['X','#eb7777',[5,0,0]],['Y','#78d18b',[0,5,0]],['Z','#9eabf0',[0,0,2]]]){const tip=project(end,w,h,scale);ctx.strokeStyle=tint;ctx.beginPath();ctx.moveTo(...origin);ctx.lineTo(...tip);ctx.stroke();ctx.fillStyle=tint;ctx.fillText(name,tip[0],tip[1]);}}
let pending=false;function redraw(){if(pending)return;pending=true;requestAnimationFrame(()=>{pending=false;canvases.forEach(draw);});}
function zoom(mult,x,y,w,h){const next=Math.max(.1,Math.min(80,view.zoom*mult)),k=next/view.zoom;view.panX=x-w/2-(x-w/2-view.panX)*k;view.panY=y-h/2-(y-h/2-view.panY)*k;view.zoom=next;}
for(const canvas of canvases){let last=null;canvas.addEventListener('pointerdown',e=>{canvas.setPointerCapture(e.pointerId);last=[e.clientX,e.clientY];});canvas.addEventListener('pointermove',e=>{if(!last)return;const dx=e.clientX-last[0],dy=e.clientY-last[1];if(e.shiftKey){view.panX+=dx;view.panY+=dy;}else{view.yaw+=dx*.008;view.pitch=Math.max(-1.55,Math.min(1.55,view.pitch+dy*.008));}redraw();});for(const event of ['pointerup','pointercancel'])canvas.addEventListener(event,()=>{last=null;});canvas.addEventListener('wheel',e=>{e.preventDefault();zoom(Math.exp(-e.deltaY*.001),e.offsetX,e.offsetY,canvas.clientWidth,canvas.clientHeight);redraw();},{passive:false});}
document.getElementById('zin').onclick=()=>{zoom(1.6,canvases[0].clientWidth/2,canvases[0].clientHeight/2,canvases[0].clientWidth,canvases[0].clientHeight);redraw();};document.getElementById('zout').onclick=()=>{zoom(1/1.6,canvases[0].clientWidth/2,canvases[0].clientHeight/2,canvases[0].clientWidth,canvases[0].clientHeight);redraw();};document.getElementById('reset').onclick=()=>{Object.assign(view,initial);redraw();};window.addEventListener('resize',redraw);redraw();
})();
</script></html>'''.replace("__DATA__", payload)
    path.write_text(markup, encoding="utf-8")


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vod-root", type=Path, required=True)
    parser.add_argument("--stage1-checkpoint", type=Path, required=True)
    parser.add_argument("--stage2-checkpoint", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--faulty-root", type=Path,
                        help="Pre-aligned N×4 float32 .bin files; omit for synthetic faults")
    parser.add_argument("--fault-cache-root", type=Path,
                        help="Severe fog cache used by the diffusion training run")
    parser.add_argument("--fault-type", choices=("auto", *FAULT_NAMES), default="auto",
                        help="Force one severe fault in every selected preview frame")
    parser.add_argument("--data-cache-root", type=Path)
    parser.add_argument("--stage2-cache-root", type=Path,
                        help="Reuse the fixed Stage-I outputs from Stage-II training")
    parser.add_argument("--radar-variant", default="radar_5frames")
    parser.add_argument("--split", choices=("train", "val", "test"), default="val")
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--random-seed", type=int, default=3103)
    parser.add_argument("--sampling-steps", type=int, default=64)
    parser.add_argument("--occupancy-threshold", type=float, default=0.5)
    parser.add_argument("--html", action="store_true",
                        help="Write an interactive faulty/clean/reconstruction comparison")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if (args.device != "cuda" or not torch.cuda.is_available() or
            args.limit < 1 or args.sampling_steps < 1 or
            not 0 < args.occupancy_threshold < 1 or
            (args.split == "test" and args.faulty_root is None) or
            (args.faulty_root is None and args.fault_cache_root is None) or
            (args.faulty_root is not None and args.fault_type != "auto")):
        parser.error("Inference requires CUDA, a severe fog cache for synthetic faults, and real faulty LiDAR for test")
    if args.stage2_cache_root:
        saved1 = torch.load(args.stage1_checkpoint, map_location="cpu",
                            weights_only=False)
        if saved1.get("radar_variant") != args.radar_variant:
            raise ValueError("Stage-I radar variant differs from the selected dataset")
        stage1 = None
    else:
        stage1, saved1 = load_stage1(args.stage1_checkpoint, args.device,
                                      args.radar_variant)
    model, saved2 = load_stage2(args.stage2_checkpoint, args.device)
    if (saved2.get("fault_policy_version") != FAULT_POLICY_VERSION
            or tuple(tuple(item) for item in saved2.get("fault_plan", ()))
            != FAULT_PLAN):
        raise ValueError("This diffusion checkpoint used the old random-dropout fault policy; retrain diffusion with the severe fault plan")
    if (saved2["stage1_sha256"] != sha256(args.stage1_checkpoint)
            or saved2["radar_variant"] != args.radar_variant):
        raise ValueError("Stage-II checkpoint was trained with a different Stage I")
    dataset = paired_dataset(args.vod_root, args.split, args.radar_variant,
                             include_clean=args.faulty_root is None,
                             data_cache_root=(args.data_cache_root
                                              if args.faulty_root is None else None))
    fog_cache = (SevereFogCache(args.fault_cache_root, args.split,
                                [frame.frame_id for frame in dataset.frames],
                                seed=int(saved2["fault_seed"]))
                 if args.faulty_root is None else None)
    cache = None
    if args.stage2_cache_root:
        expected = cache_identity(args.stage1_checkpoint, args.vod_root,
                                  args.radar_variant,
                                  float(saved1.get("radius_m", 3.2)),
                                  float(saved2["gate_threshold"]), args.split,
                                  [frame.frame_id for frame in dataset.frames])
        cache = Stage2FixedCache(args.stage2_cache_root / args.split, expected)
        cache.check_complete()
    indices = random.Random(args.random_seed).sample(
        range(len(dataset)), min(args.limit, len(dataset)))
    args.output_root.mkdir(parents=True, exist_ok=True)
    alpha_bars = cosine_alpha_bars(device=args.device)
    summary = []
    for index in indices:
        row = dataset[index]
        frame_id = row["frame_id"]
        if cache is None:
            radar = row["radar"].to(args.device)
            output = stage1_frame(stage1, radar, float(saved1.get("radius_m", 3.2)))
            targets = None
        else:
            output, targets = cache.load(frame_id, args.device)
        conditions = model.conditions(output, float(saved2["gate_threshold"]))
        if not len(conditions.hierarchy[0]):
            print(f"{frame_id}: no radar-allowed cells; skipped", flush=True)
            continue
        if args.faulty_root:
            faulty, fault_type = load_faulty(args.faulty_root, args.split, frame_id), "real"
        else:
            faulty, fault_type = severe_fault(
                row["clean_lidar"], frame_id, 0, fog_cache,
                seed=int(saved2["fault_seed"]),
                fault_type=None if args.fault_type == "auto" else args.fault_type)
        faulty = faulty.to(args.device)
        seed = int.from_bytes(hashlib.sha256(
            f"{args.random_seed}:{frame_id}".encode()).digest()[:8], "little")
        torch.manual_seed(seed)
        predicted = model.sample(faulty, conditions, alpha_bars,
                                 steps=args.sampling_steps,
                                 occupancy_threshold=args.occupancy_threshold)
        xyz = predicted["points_xyz"].cpu().numpy().astype(np.float32)
        scores = predicted["probabilities"].cpu().numpy().astype(np.float32)
        faulty_roi = _roi(faulty.cpu().numpy())
        clean_roi = (_roi(row["clean_lidar"].numpy())
                     if "clean_lidar" in row else np.empty((0, 3), dtype=np.float32))
        synthetic = args.faulty_root is None
        shared_subset = fault_type in ("total_loss", "fov_filter")
        changed_geometry = changed_reflectivity = None
        if synthetic:
            kept = _present_in_faulty(clean_roi, faulty_roi)
            xyz_kept = _present_in_faulty(clean_roi[:, :3], faulty_roi[:, :3])
            changed_geometry = int((~xyz_kept).sum())
            changed_reflectivity = int((xyz_kept & ~kept).sum())
        clean_display = clean_roi
        if shared_subset:
            display_kept = _present_in_faulty(clean_display, faulty_roi)
            faulty_display = clean_display[display_kept, :3]
            removed_display = clean_display[~display_kept, :3]
        else:
            faulty_display = faulty_roi[:, :3]
            removed_display = (clean_display[~_present_in_faulty(
                clean_display, faulty_roi), :3] if synthetic
                else np.empty((0, 3), dtype=np.float32))
        median_intensity = (
            float(np.median(clean_roi[:, 3])) if len(clean_roi) else None,
            float(np.median(faulty_roi[:, 3])) if len(faulty_roi) else None)
        stem_name = (f"{frame_id}_{fault_type}_stage2_epoch_{saved2['epoch']:03d}"
                     if args.html else f"{frame_id}_stage2")
        stem = args.output_root / stem_name
        _write_ply(stem.with_suffix(".ply"), xyz, scores)
        np.savez_compressed(stem.with_suffix(".npz"), xyz=xyz,
                            probability=scores,
                            coordinates=predicted["coordinates"].cpu().numpy(),
                            occupied_coordinates=predicted[
                                "occupied_coordinates"].cpu().numpy(),
                            point_counts=predicted["point_counts"].cpu().numpy(),
                            faulty_xyz=faulty_roi[:, :3], clean_xyz=clean_roi[:, :3],
                            faulty_reflectivity=faulty_roi[:, 3],
                            clean_reflectivity=clean_roi[:, 3],
                            fault_type=np.asarray(fault_type))
        stats = {"frame_id": frame_id, "epoch": int(saved2["epoch"]),
                 "fault_type": fault_type,
                 "allowed_cells": predicted["allowed_cells"],
                 "predicted_occupied_cells": predicted["predicted_cells"],
                 "reconstructed_points": predicted["predicted_points"],
                 "max_points_in_one_predicted_cell": (
                     int(predicted["point_counts"].max())
                     if predicted["predicted_cells"] else 0),
                 "faulty_input_points": len(faulty_roi),
                 "clean_reference_points": len(clean_roi),
                 "clean_points_absent_or_shifted": changed_geometry,
                 "clean_points_reflectivity_changed": changed_reflectivity,
                 "median_clean_reflectivity": median_intensity[0],
                 "median_faulty_reflectivity": median_intensity[1]}
        metrics = None
        if "clean_lidar" in row:
            if targets is None:
                clean_for_targets = row["clean_lidar"][:, :3].to(args.device)
                targets = reconstruction_targets(clean_for_targets,
                                                 conditions.hierarchy[0])
            occupied, known_free, _, coverage = targets
            allowed_keys = encode_keys(conditions.hierarchy[0], GRID.shape_zyx)
            chosen_keys = encode_keys(predicted["occupied_coordinates"], GRID.shape_zyx)
            chosen_positions = torch.searchsorted(allowed_keys, chosen_keys)
            selected = torch.zeros(len(allowed_keys), dtype=torch.bool,
                                   device=args.device)
            selected[chosen_positions] = True
            intersection = (selected & occupied).sum()
            known = occupied | known_free
            known_union = ((selected & known) | occupied).sum().clamp_min(1)
            full_union = (selected | occupied).sum().clamp_min(1)
            metrics = {"known_iou": float(intersection / known_union),
                       "full_iou": float(intersection / full_union),
                       "gate_coverage": coverage}
            stats.update(metrics)
        if args.html:
            viewer = stem.with_suffix(".html")
            _write_html(viewer, frame_id=frame_id, epoch=int(saved2["epoch"]),
                        split=args.split,
                        faulty=faulty_display, clean=clean_display[:, :3],
                        predicted=xyz,
                        removed=removed_display,
                        threshold=args.occupancy_threshold,
                        allowed_cells=predicted["allowed_cells"],
                        predicted_cells=predicted["predicted_cells"],
                        sampling_steps=args.sampling_steps,
                        fault_type=fault_type,
                        totals=(len(faulty_roi), len(clean_roi), len(xyz)),
                        changed_geometry=changed_geometry,
                        changed_reflectivity=changed_reflectivity,
                        median_intensity=median_intensity,
                        shared_subset=shared_subset, metrics=metrics)
            stats["viewer"] = viewer.name
        summary.append(stats)
        print(json.dumps(stats), flush=True)
    if not summary:
        raise RuntimeError("Selected frames had no radar-allowed cells")
    (args.output_root / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")
    if args.html:
        links = "\n".join(
            f'<li><a target="viewer" href="{html.escape(item["viewer"])}">'
            f'Frame {html.escape(item["frame_id"])} · {html.escape(item["fault_type"])} · epoch {item["epoch"]}</a> · '
            f'{item["reconstructed_points"]:,} reconstructed'
            + (f'; {item["clean_points_absent_or_shifted"]:,} XYZ changed, '
               f'{item["clean_points_reflectivity_changed"]:,} reflectivity changed'
               if item["clean_points_absent_or_shifted"] is not None else '')
            + '</li>'
            for item in summary)
        first = html.escape(summary[0]["viewer"])
        index = ("<!doctype html><html lang=\"en\"><meta charset=\"utf-8\">"
                 "<title>Stage II diffusion samples</title>"
                 "<style>body{margin:1rem;background:#101820;color:#edf4fa;font:16px system-ui}"
                 "a{color:#88ceff}iframe{width:100%;height:85vh;border:1px solid #526474}</style>"
                 f"<h1>Stage II diffusion {html.escape(args.split)} reconstructions</h1>"
                 f"<ul>{links}</ul><iframe name=\"viewer\" src=\"{first}\" "
                 "title=\"Diffusion reconstruction\"></iframe></html>")
        (args.output_root / "index.html").write_text(index, encoding="utf-8")
        print(f"Open {args.output_root / 'index.html'}", flush=True)


if __name__ == "__main__":
    main()

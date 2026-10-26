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
from models.radar_lidar_stage2.sparse_geometry import (
    GRID, reconstruction_targets, voxelize_means,
)
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
    """Match exact XYZ rows."""
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
                faulty: np.ndarray, clean: np.ndarray, radar: np.ndarray,
                predicted: np.ndarray, probabilities: np.ndarray,
                threshold: float, allowed_cells: int, predicted_cells: int,
                sampling_steps: int,
                fault_type: str, totals: tuple[int, int, int, int],
                removed_clean_points: int | None,
                full_estimate_cells: int, overlap_observed_cells: int,
                missing_only: bool, zero_residual: bool,
                metrics: dict | None) -> None:
    if len(predicted) != len(probabilities):
        raise ValueError("One confidence value is required per reconstructed point")
    clouds = [faulty, clean, radar, predicted]
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
        "radar": np.round(radar, 3).tolist(),
        "predicted": np.round(predicted, 3).tolist(),
        "probabilities": np.round(probabilities.astype(np.float64), 4).tolist(),
        "threshold": threshold, "allowed_cells": allowed_cells,
        "predicted_cells": predicted_cells,
        "sampling_steps": sampling_steps, "fault_type": fault_type,
        "totals": totals, "removed_clean_points": removed_clean_points,
        "full_estimate_cells": full_estimate_cells,
        "overlap_observed_cells": overlap_observed_cells,
        "missing_only": missing_only,
        "zero_residual": zero_residual,
        "metrics": metrics,
        "minimum": minimum, "maximum": maximum,
    }, separators=(",", ":")).replace("<", "\\u003c")
    path.write_text(_viewer_markup(payload), encoding="utf-8")


def _viewer_markup(payload: str) -> str:
    markup = r'''<!doctype html>
<html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Stage II diffusion reconstruction</title>
<style>
body{margin:0;padding:12px;background:#101820;color:#eef5fa;font:14px system-ui,sans-serif}
h1{font-size:20px;margin:0 0 6px}p{color:#b9c8d6;margin:5px 0 10px}
.controls{display:flex;gap:10px;align-items:center;flex-wrap:wrap;margin:12px 0}button{background:#253847;color:inherit;border:1px solid #637b8b;border-radius:4px;padding:5px 9px;cursor:pointer}
.panels{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:8px}
section{min-width:0;background:#182633;border:1px solid #3b5164;border-radius:5px;padding:6px}
h2{font-size:14px;margin:0 0 5px}canvas{width:100%;height:48vh;min-height:340px;background:#0c1620;touch-action:none;cursor:grab}
.layers{display:flex;gap:4px 12px;flex-wrap:wrap;margin:5px 0 8px}.layers label{white-space:nowrap;cursor:pointer}.layers input{vertical-align:middle}
.confidence,.camera-speed,.pitch-control,.rotation-mode{display:flex;align-items:center;gap:7px;flex-wrap:wrap}.confidence input{width:220px}.camera-speed input,.pitch-control input{width:130px}select{background:#253847;color:inherit;border:1px solid #637b8b;border-radius:4px;padding:4px}
.legend{color:#b9c8d6;margin-top:8px}
@media(max-width:750px){.panels{display:block}section{margin-bottom:8px}canvas{height:50vh;min-height:280px}}
</style>
<h1 id="title"></h1><p id="note"></p><p id="counts"></p>
<div class="controls"><button id="zin">Zoom in</button><button id="zout">Zoom out</button><button id="reset">Reset view</button><label class="rotation-mode">Drag rotation <select id="rotation-mode"><option value="locked">Axis locked</option><option value="pitch">Pitch only</option><option value="yaw">Yaw only</option><option value="free">Free</option></select></label><output id="yaw-value">Yaw -37°</output><label class="pitch-control">Pitch <output id="pitch-value">20°</output><input id="pitch-angle" type="range" min="-89" max="89" step="1" value="20"></label><label class="camera-speed">Camera speed <output id="camera-speed-value">0.7×</output><input id="camera-speed" type="range" min="0.2" max="2" step="0.1" value="0.7"></label><label class="confidence">Reconstruction confidence ≥ <output id="confidence-value"></output><input id="confidence" type="range" max="1" step="0.01"></label></div>
<div id="panels" class="panels"></div>
<p class="legend"><span style="color:#ffb85c">●</span> faulty input &nbsp; <span style="color:#60d796">●</span> clean reference &nbsp; <span style="color:#67b7ff">●</span> aligned radar &nbsp; <span style="color:#ff8b6c">●</span> reconstruction. Each panel has its own layer switches. The confidence slider filters reconstructed points in every panel; it cannot show points below the generation threshold. Vertical drag pitches and horizontal drag yaws; the default drag mode locks to one axis. Choose Pitch only or use the Pitch slider to change elevation without yaw. Shift-drag or right-drag pans; the wheel zooms. All panels share one camera. PLY and NPZ contain the complete reconstruction.</p>
<script id="data" type="application/json">__DATA__</script>
<script>
(() => {
const d=JSON.parse(document.getElementById('data').textContent);
document.getElementById('title').textContent=`${d.split} frame ${d.frame} · ${d.fault_type} · diffusion epoch ${d.epoch}`;
const explanations={total_loss:'All LiDAR returns are removed.',fov_filter:'The established FOV filter at severity 5 retains only its narrowest angular sector.',fog_sim:'Fog-response returns are removed; retained input points have original clean XYZ. LiDAR intensity is not a model input.',real:'Pre-aligned real faulty LiDAR was loaded; its intensity column is ignored.'};
document.getElementById('note').textContent=`${explanations[d.fault_type]} ${d.zero_residual?'Zero-residual ablation: the faulty LiDAR latent goes directly to LPCR; radar still selects the allowed region.':'The model receives faulty LiDAR and radar features; clean LiDAR is used only as the reference.'} ${d.missing_only?'The reconstructed layer contains additions only; faulty-occupied voxels are preserved from the input.':'The reconstructed layer shows the unfiltered full-cloud estimate.'}`;
document.getElementById('counts').textContent=`${d.totals[0].toLocaleString()} faulty points, ${d.totals[1].toLocaleString()} clean reference points, ${d.totals[2].toLocaleString()} aligned radar points, ${d.totals[3].toLocaleString()} shown reconstructed points. The full estimate predicted ${d.full_estimate_cells.toLocaleString()} occupied voxels; ${d.overlap_observed_cells.toLocaleString()} coincide with faulty-occupied voxels. ${d.removed_clean_points===null?'Clean-point difference unavailable for real faulty LiDAR':`${d.removed_clean_points.toLocaleString()} clean points removed by the synthetic fault.`} ${d.allowed_cells.toLocaleString()} radar-allowed cells; generation threshold ${d.threshold.toFixed(2)}; ${d.sampling_steps} diffusion steps.`;
if(d.metrics){const m=d.metrics,p=document.createElement('p'),missing=m.missing_precision==null?'':`missing-voxel precision ${m.missing_precision.toFixed(4)}, recall ${m.missing_recall.toFixed(4)}, F1 ${m.missing_f1.toFixed(4)}, `;p.textContent=`${m.metric_scope||'Metrics'}: Known IoU ${m.known_iou.toFixed(4)}, Full IoU ${m.full_iou.toFixed(4)}, ${missing}gate coverage ${m.gate_coverage.toFixed(4)}. The display slider does not recompute these metrics.`;document.getElementById('counts').after(p);}
const layers=[{key:'faulty',name:'Faulty LiDAR',color:'#ffb85c'},{key:'clean',name:'Clean LiDAR',color:'#60d796'},{key:'radar',name:'Radar',color:'#67b7ff'},{key:'predicted',name:'Reconstructed',color:'#ff8b6c'}];
const presets=[{name:'Faulty LiDAR',visible:['faulty']},{name:'Clean LiDAR',visible:['clean']},{name:'Reconstructed + radar',visible:['predicted','radar']},{name:'Faulty LiDAR + radar + reconstructed',visible:['faulty','radar','predicted']}];
const panes=document.getElementById('panels'),panels=[];
for(const preset of presets){const section=document.createElement('section'),h=document.createElement('h2'),switches=document.createElement('div'),canvas=document.createElement('canvas'),inputs={};switches.className='layers';h.textContent=preset.name;for(const layer of layers){const label=document.createElement('label'),input=document.createElement('input'),swatch=document.createElement('span');input.type='checkbox';input.checked=preset.visible.includes(layer.key);input.setAttribute('aria-label',`${preset.name}: ${layer.name}`);input.addEventListener('change',redraw);swatch.textContent='● ';swatch.style.color=layer.color;label.append(input,swatch,document.createTextNode(layer.name));switches.append(label);inputs[layer.key]=input;}section.append(h,switches,canvas);panes.append(section);panels.push({name:preset.name,heading:h,canvas,inputs});}
const confidence=document.getElementById('confidence'),confidenceValue=document.getElementById('confidence-value');confidence.min=String(d.threshold);confidence.value=String(d.threshold);confidenceValue.textContent=Number(confidence.value).toFixed(2);
const cameraSpeed=document.getElementById('camera-speed'),cameraSpeedValue=document.getElementById('camera-speed-value');cameraSpeed.addEventListener('input',()=>{cameraSpeedValue.textContent=`${Number(cameraSpeed.value).toFixed(1)}×`;});
const rotationMode=document.getElementById('rotation-mode'),pitchAngle=document.getElementById('pitch-angle'),pitchValue=document.getElementById('pitch-value'),yawValue=document.getElementById('yaw-value');
const mid=d.minimum.map((x,i)=>(x+d.maximum[i])/2),extent=Math.hypot(...d.minimum.map((x,i)=>d.maximum[i]-x))||1;
const initial={yaw:-.65,pitch:.35,zoom:1,panX:0,panY:0},view={...initial},maxPitch=89*Math.PI/180;
function project(p,w,h,s){const x=p[0]-mid[0],y=p[1]-mid[1],z=p[2]-mid[2],cy=Math.cos(view.yaw),sy=Math.sin(view.yaw),cp=Math.cos(view.pitch),sp=Math.sin(view.pitch);const u=cy*x-sy*y,v=sy*x+cy*y;return [w/2+view.panX+u*s,h/2+view.panY-(cp*z-sp*v)*s];}
function plot(ctx,points,w,h,s,color,scores,cutoff,size){let selected=0;ctx.fillStyle=color;for(let i=0;i<points.length;i++){if(scores&&scores[i]<cutoff)continue;selected++;const [x,y]=project(points[i],w,h,s);if(x>=0&&x<w&&y>=0&&y<h)ctx.fillRect(x-size/2,y-size/2,size,size);}return selected;}
function draw(panel){const canvas=panel.canvas,ratio=devicePixelRatio||1,w=canvas.clientWidth,h=canvas.clientHeight;if(!w||!h)return;canvas.width=Math.round(w*ratio);canvas.height=Math.round(h*ratio);const ctx=canvas.getContext('2d');ctx.setTransform(ratio,0,0,ratio,0,0);ctx.clearRect(0,0,w,h);const scale=.88*Math.min(w,h)*view.zoom/extent;let count=0;for(const key of ['predicted','faulty','clean','radar']){if(!panel.inputs[key].checked)continue;const layer=layers.find(item=>item.key===key);count+=plot(ctx,d[key],w,h,scale,layer.color,key==='predicted'?d.probabilities:null,Number(confidence.value),key==='radar'?4:2);}panel.heading.textContent=`${panel.name}: ${count.toLocaleString()} layer points selected`;const origin=project([0,0,0],w,h,scale);for(const [name,tint,end] of [['X','#eb7777',[5,0,0]],['Y','#78d18b',[0,5,0]],['Z','#9eabf0',[0,0,2]]]){const tip=project(end,w,h,scale);ctx.strokeStyle=tint;ctx.beginPath();ctx.moveTo(...origin);ctx.lineTo(...tip);ctx.stroke();ctx.fillStyle=tint;ctx.fillText(name,tip[0],tip[1]);}}
let pending=false;function redraw(){pitchAngle.value=String(Math.round(view.pitch*180/Math.PI));pitchValue.textContent=`${Number(pitchAngle.value)}°`;yawValue.textContent=`Yaw ${Math.round(view.yaw*180/Math.PI)}°`;if(pending)return;pending=true;requestAnimationFrame(()=>{pending=false;panels.forEach(draw);});}
function zoom(mult,x,y,w,h){const next=Math.max(.1,Math.min(80,view.zoom*mult)),k=next/view.zoom;view.panX=x-w/2-(x-w/2-view.panX)*k;view.panY=y-h/2-(y-h/2-view.panY)*k;view.zoom=next;}
for(const {canvas} of panels){
  let drag=null;
  canvas.addEventListener('contextmenu',e=>e.preventDefault());
  canvas.addEventListener('pointerdown',e=>{canvas.setPointerCapture(e.pointerId);drag={x:e.clientX,y:e.clientY,startX:e.clientX,startY:e.clientY,axis:null,pan:e.button===2};canvas.style.cursor='grabbing';});
  canvas.addEventListener('pointermove',e=>{
    if(!drag)return;
    let dx=e.clientX-drag.x,dy=e.clientY-drag.y;
    const speed=Number(cameraSpeed.value),mode=rotationMode.value;
    if(!e.shiftKey&&!drag.pan&&mode==='locked'&&!drag.axis){
      const totalX=e.clientX-drag.startX,totalY=e.clientY-drag.startY;
      if(Math.hypot(totalX,totalY)<10)return;
      drag.axis=Math.abs(totalY)>=Math.abs(totalX)?'pitch':'yaw';
      dx=totalX;dy=totalY;
    }
    drag.x=e.clientX;drag.y=e.clientY;
    if(e.shiftKey||drag.pan){drag.startX=e.clientX;drag.startY=e.clientY;view.panX+=dx*speed;view.panY+=dy*speed;}
    else{
      const axis=mode==='locked'?drag.axis:mode;
      if(axis==='yaw'||axis==='free')view.yaw+=dx*.004*speed;
      if(axis==='pitch'||axis==='free')view.pitch=Math.max(-maxPitch,Math.min(maxPitch,view.pitch+dy*.004*speed));
    }
    redraw();
  });
  for(const event of ['pointerup','pointercancel','lostpointercapture'])canvas.addEventListener(event,()=>{drag=null;canvas.style.cursor='grab';});
  canvas.addEventListener('wheel',e=>{e.preventDefault();zoom(Math.exp(-e.deltaY*.0006*Number(cameraSpeed.value)),e.offsetX,e.offsetY,canvas.clientWidth,canvas.clientHeight);redraw();},{passive:false});
}
pitchAngle.addEventListener('input',()=>{view.pitch=Number(pitchAngle.value)*Math.PI/180;redraw();});
confidence.addEventListener('input',()=>{confidenceValue.textContent=Number(confidence.value).toFixed(2);redraw();});
document.getElementById('zin').onclick=()=>{const canvas=panels[0].canvas;zoom(1.25,canvas.clientWidth/2,canvas.clientHeight/2,canvas.clientWidth,canvas.clientHeight);redraw();};document.getElementById('zout').onclick=()=>{const canvas=panels[0].canvas;zoom(1/1.25,canvas.clientWidth/2,canvas.clientHeight/2,canvas.clientWidth,canvas.clientHeight);redraw();};document.getElementById('reset').onclick=()=>{Object.assign(view,initial);redraw();};window.addEventListener('resize',redraw);redraw();
})();
</script></html>'''
    return markup.replace("__DATA__", payload)


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
    parser.add_argument("--full-estimate", action="store_true",
                        help="Show all predicted occupied voxels, including those already observed by faulty LiDAR")
    parser.add_argument("--zero-residual", action="store_true",
                        help="Ablation: decode faulty LiDAR latent without diffusion residual")
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
                                 occupancy_threshold=args.occupancy_threshold,
                                 missing_only=not args.full_estimate,
                                 use_residual=not args.zero_residual)
        xyz = predicted["points_xyz"].cpu().numpy().astype(np.float32)
        scores = predicted["probabilities"].cpu().numpy().astype(np.float32)
        faulty_roi = _roi(faulty.cpu().numpy())
        radar_roi = _roi(row["radar"].cpu().numpy())
        clean_roi = (_roi(row["clean_lidar"][:, :3].numpy())
                     if "clean_lidar" in row else np.empty((0, 3), dtype=np.float32))
        synthetic = args.faulty_root is None
        removed_clean_points = None
        if synthetic:
            if len(faulty_roi) > len(clean_roi):
                raise RuntimeError("Synthetic fault added LiDAR returns")
            removed_clean_points = len(clean_roi) - len(faulty_roi)
        stem_name = (f"{frame_id}_{fault_type}_stage2_epoch_{saved2['epoch']:03d}"
                     if args.html else f"{frame_id}_stage2")
        if args.zero_residual:
            stem_name += "_zero_residual"
        if args.full_estimate:
            stem_name += "_full_estimate"
        stem = args.output_root / stem_name
        _write_ply(stem.with_suffix(".ply"), xyz, scores)
        if not args.full_estimate:
            completed_xyz = np.concatenate((faulty_roi[:, :3], xyz))
            completed_scores = np.concatenate((np.ones(len(faulty_roi), dtype=np.float32),
                                               scores))
            _write_ply(stem.with_name(stem.name + "_completed.ply"),
                       completed_xyz, completed_scores)
        np.savez_compressed(stem.with_suffix(".npz"), xyz=xyz,
                            probability=scores,
                            coordinates=predicted["coordinates"].cpu().numpy(),
                            occupied_coordinates=predicted[
                                "occupied_coordinates"].cpu().numpy(),
                            faulty_xyz=faulty_roi[:, :3], clean_xyz=clean_roi[:, :3],
                            radar_xyz=radar_roi[:, :3],
                            fault_type=np.asarray(fault_type),
                            residual_mode=np.asarray("zero" if args.zero_residual
                                                     else "diffusion"),
                            prediction_mode=np.asarray("full_estimate" if args.full_estimate
                                                       else "missing_only"))
        stats = {"frame_id": frame_id, "epoch": int(saved2["epoch"]),
                 "fault_type": fault_type,
                 "allowed_cells": predicted["allowed_cells"],
                 "predicted_occupied_cells": predicted["predicted_cells"],
                 "reconstructed_points": predicted["predicted_points"],
                 "completed_points": (len(faulty_roi) + predicted["predicted_points"]
                                      if not args.full_estimate else None),
                 "full_estimate_cells": predicted["full_estimate_cells"],
                 "overlap_observed_cells": predicted["overlap_observed_cells"],
                 "prediction_mode": "full_estimate" if args.full_estimate else "missing_only",
                 "residual_mode": "zero" if args.zero_residual else "diffusion",
                 "residual_abs_present_latent": predicted["residual_abs_present_latent"],
                 "residual_abs_absent_latent": predicted["residual_abs_absent_latent"],
                 "faulty_input_points": len(faulty_roi),
                 "radar_input_points": len(radar_roi),
                 "clean_reference_points": len(clean_roi),
                 "clean_points_removed": removed_clean_points}
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
            faulty_coords, _ = voxelize_means(faulty[:, :3], grid=GRID)
            faulty_keys = encode_keys(faulty_coords, GRID.shape_zyx)
            observed = torch.isin(allowed_keys, faulty_keys)
            additions = selected & ~observed
            completed = observed | additions
            scored = selected if args.full_estimate else completed
            intersection = (scored & occupied).sum()
            known = occupied | known_free
            known_union = ((scored & known) | occupied).sum().clamp_min(1)
            full_union = (scored | occupied).sum().clamp_min(1)
            missing_target = occupied & ~observed
            missing_tp = (additions & missing_target).sum()
            precision = float(missing_tp / additions.sum().clamp_min(1))
            recall = float(missing_tp / missing_target.sum().clamp_min(1))
            metrics = {"known_iou": float(intersection / known_union),
                       "full_iou": float(intersection / full_union),
                       "missing_precision": precision,
                       "missing_recall": recall,
                       "missing_f1": (2 * precision * recall /
                                      (precision + recall + 1e-8)),
                       "missing_target_cells": int(missing_target.sum()),
                       "metric_scope": ("Raw full estimate" if args.full_estimate
                                        else "Faulty plus reconstructed additions"),
                       "gate_coverage": coverage}
            stats.update(metrics)
        if args.html:
            viewer = stem.with_suffix(".html")
            _write_html(viewer, frame_id=frame_id, epoch=int(saved2["epoch"]),
                        split=args.split,
                        faulty=faulty_roi[:, :3], clean=clean_roi[:, :3],
                        radar=radar_roi[:, :3], predicted=xyz,
                        probabilities=scores,
                        threshold=args.occupancy_threshold,
                        allowed_cells=predicted["allowed_cells"],
                        predicted_cells=predicted["predicted_cells"],
                        sampling_steps=0 if args.zero_residual else args.sampling_steps,
                        fault_type=fault_type,
                        totals=(len(faulty_roi), len(clean_roi),
                                len(radar_roi), len(xyz)),
                        removed_clean_points=removed_clean_points,
                        full_estimate_cells=predicted["full_estimate_cells"],
                        overlap_observed_cells=predicted["overlap_observed_cells"],
                        missing_only=not args.full_estimate,
                        zero_residual=args.zero_residual,
                        metrics=metrics)
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
            + (f'; {item["clean_points_removed"]:,} clean points removed'
               if item["clean_points_removed"] is not None else '')
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

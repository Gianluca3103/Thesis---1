"""View clean-input Stage-II LPCR reconstructions on random VoD frames.

The Stage-I radar map selects active sites, but the LPCR encoder reads clean
LiDAR. This visualizes autoencoder pretraining, not faulty-to-clean inference.
"""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
import random

import numpy as np
import torch

from models.radar_lidar_crossmodal_v1.pipeline_data import paired_dataset
from models.radar_lidar_stage2.sparse_geometry import GRID
from models.radar_lidar_stage2.stage2_cache import Stage2FixedCache, cache_identity
from models.radar_lidar_stage2.stage2_io import load_stage2, sha256


def _write_ply(path: Path, xyz: np.ndarray, confidence: np.ndarray | None = None) -> None:
    fields = [(axis, "<f4") for axis in "xyz"]
    if confidence is not None:
        fields.append(("confidence", "<f4"))
    records = np.empty(len(xyz), dtype=fields)
    for column, axis in enumerate("xyz"):
        records[axis] = xyz[:, column]
    if confidence is not None:
        records["confidence"] = confidence
    header = ("ply\nformat binary_little_endian 1.0\n"
              f"element vertex {len(xyz)}\n"
              "property float x\nproperty float y\nproperty float z\n"
              + ("property float confidence\n" if confidence is not None else "")
              + "end_header\n")
    with path.open("wb") as stream:
        stream.write(header.encode("ascii"))
        stream.write(records.tobytes())


def _display_subset(xyz: np.ndarray, scores: np.ndarray | None, limit: int,
                    seed: int) -> tuple[np.ndarray, np.ndarray | None]:
    if len(xyz) <= limit:
        return xyz, scores
    indices = np.sort(np.random.default_rng(seed).choice(len(xyz), limit, replace=False))
    return xyz[indices], None if scores is None else scores[indices]


def _write_html(path: Path, frame_id: str, split: str, epoch: int, clean: np.ndarray,
                reconstructed: np.ndarray, scores: np.ndarray,
                threshold: float, clean_total: int, candidate_total: int) -> None:
    all_xyz = np.concatenate((clean, reconstructed), axis=0)
    if len(all_xyz):
        minimum, maximum = all_xyz.min(0).tolist(), all_xyz.max(0).tolist()
    else:
        minimum, maximum = list(GRID.minimum_xyz), list(GRID.maximum_xyz)
    payload = json.dumps({
        "frame": frame_id, "split": split, "epoch": epoch,
        "clean": np.round(clean, 3).tolist(),
        "reconstructed": np.round(reconstructed, 3).tolist(),
        "scores": np.round(scores, 4).tolist(),
        "threshold": threshold, "clean_total": clean_total,
        "candidate_total": candidate_total,
        "minimum": minimum, "maximum": maximum,
    }, separators=(",", ":")).replace("<", "\\u003c")
    markup = r'''<!doctype html>
<html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Stage II LPCR reconstruction</title>
<style>
body{margin:0;padding:12px;background:#101820;color:#eef5fa;font:14px system-ui,sans-serif}
h1{font-size:20px;margin:0 0 6px}p{color:#b9c8d6;margin:5px 0 10px}
.controls{display:flex;gap:14px;align-items:center;flex-wrap:wrap;margin:12px 0}
input[type=range]{width:250px}button{background:#253847;color:inherit;border:1px solid #637b8b;border-radius:4px;padding:5px 9px;cursor:pointer}
.panels{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:8px}
section{min-width:0;background:#182633;border:1px solid #3b5164;border-radius:5px;padding:6px}
h2{font-size:14px;margin:0 0 5px}canvas{width:100%;height:70vh;min-height:420px;background:#0c1620;touch-action:none;cursor:grab}
.legend{color:#b9c8d6;margin-top:8px}
@media(max-width:900px){.panels{display:block}section{margin-bottom:8px}canvas{height:55vh;min-height:300px}}
</style>
<h1 id="title"></h1>
<p>Clean LiDAR is the encoder input for this LPCR autoencoder. Radar determines the allowed cells through the Stage I map. This is not a faulty-LiDAR reconstruction.</p>
<p id="counts"></p>
<div class="controls"><label>LPCR occupancy threshold <input id="threshold" type="range" min="0" max="1" step="0.01"><output id="value"></output></label>
<button id="zin">Zoom in</button><button id="zout">Zoom out</button><button id="reset">Reset view</button></div>
<div id="panels" class="panels"></div>
<p class="legend"><span style="color:#60d796">●</span> clean LiDAR &nbsp; <span style="color:#ff8b6c">●</span> reconstructed LPCR points. Drag to rotate; Shift-drag to pan; wheel to zoom. All panels share one camera. PLY and NPZ files contain the complete data.</p>
<script id="data" type="application/json">__DATA__</script>
<script>
(() => {
const data=JSON.parse(document.getElementById('data').textContent);
document.getElementById('title').textContent=`${data.split} frame ${data.frame} · LPCR epoch ${data.epoch}`;
document.getElementById('counts').textContent=`${data.clean_total.toLocaleString()} clean LiDAR points in the Stage II ROI; ${data.candidate_total.toLocaleString()} radar-allowed cells. The browser displays ${data.clean.length.toLocaleString()} clean points and ${data.reconstructed.length.toLocaleString()} candidate reconstructions (sampled if needed).`;
const slider=document.getElementById('threshold'),value=document.getElementById('value');slider.value=data.threshold;value.textContent=Number(slider.value).toFixed(2);
const names=['Clean LiDAR','LPCR reconstruction','Overlay'],panes=document.getElementById('panels'),canvases=[],headings=[];
for(const name of names){const section=document.createElement('section'),h=document.createElement('h2'),canvas=document.createElement('canvas');h.textContent=name;section.append(h,canvas);panes.append(section);canvases.push(canvas);headings.push(h);}
const mid=data.minimum.map((x,i)=>(x+data.maximum[i])/2),extent=Math.hypot(...data.minimum.map((x,i)=>data.maximum[i]-x))||1;
const initial={yaw:-.65,pitch:.35,zoom:1,panX:0,panY:0},view={...initial};
function project(p,w,h,s){const x=p[0]-mid[0],y=p[1]-mid[1],z=p[2]-mid[2],cy=Math.cos(view.yaw),sy=Math.sin(view.yaw),cp=Math.cos(view.pitch),sp=Math.sin(view.pitch);const u=cy*x-sy*y,v=sy*x+cy*y;return [w/2+view.panX+u*s,h/2+view.panY-(cp*z-sp*v)*s];}
function plot(ctx,points,w,h,s,color,cutoff){let count=0;ctx.fillStyle=color;for(let i=0;i<points.length;i++){if(cutoff!==null&&data.scores[i]<cutoff)continue;count++;const [x,y]=project(points[i],w,h,s);if(x>=0&&x<w&&y>=0&&y<h)ctx.fillRect(x-1,y-1,2,2);}return count;}
function draw(canvas,index){const ratio=devicePixelRatio||1,w=canvas.clientWidth,h=canvas.clientHeight;if(!w||!h)return;canvas.width=Math.round(w*ratio);canvas.height=Math.round(h*ratio);const ctx=canvas.getContext('2d');ctx.setTransform(ratio,0,0,ratio,0,0);ctx.clearRect(0,0,w,h);const scale=.88*Math.min(w,h)*view.zoom/extent,cutoff=Number(slider.value);let count=0;if(index!==1)plot(ctx,data.clean,w,h,scale,'#60d796',null);if(index!==0)count=plot(ctx,data.reconstructed,w,h,scale,'#ff8b6c',cutoff);headings[index].textContent=names[index]+(index===0?`: ${data.clean.length.toLocaleString()} shown`:`: ${count.toLocaleString()} predicted points`);const origin=project([0,0,0],w,h,scale);for(const [name,tint,end] of [['X','#eb7777',[5,0,0]],['Y','#78d18b',[0,5,0]],['Z','#9eabf0',[0,0,2]]]){const tip=project(end,w,h,scale);ctx.strokeStyle=tint;ctx.beginPath();ctx.moveTo(...origin);ctx.lineTo(...tip);ctx.stroke();ctx.fillStyle=tint;ctx.fillText(name,tip[0],tip[1]);}}
let pending=false;function redraw(){if(pending)return;pending=true;requestAnimationFrame(()=>{pending=false;canvases.forEach(draw);});}
function zoom(mult,x,y,w,h){const next=Math.max(.1,Math.min(80,view.zoom*mult)),k=next/view.zoom;view.panX=x-w/2-(x-w/2-view.panX)*k;view.panY=y-h/2-(y-h/2-view.panY)*k;view.zoom=next;}
for(const canvas of canvases){let last=null;canvas.addEventListener('pointerdown',e=>{canvas.setPointerCapture(e.pointerId);last=[e.clientX,e.clientY];});canvas.addEventListener('pointermove',e=>{if(!last)return;const dx=e.clientX-last[0],dy=e.clientY-last[1];if(e.shiftKey){view.panX+=dx;view.panY+=dy;}else{view.yaw+=dx*.008;view.pitch=Math.max(-1.55,Math.min(1.55,view.pitch+dy*.008));}redraw();});for(const event of ['pointerup','pointercancel'])canvas.addEventListener(event,()=>{last=null;});canvas.addEventListener('wheel',e=>{e.preventDefault();zoom(Math.exp(-e.deltaY*.001),e.offsetX,e.offsetY,canvas.clientWidth,canvas.clientHeight);redraw();},{passive:false});}
slider.addEventListener('input',()=>{value.textContent=Number(slider.value).toFixed(2);redraw();});document.getElementById('zin').onclick=()=>{zoom(1.6,canvases[0].clientWidth/2,canvases[0].clientHeight/2,canvases[0].clientWidth,canvases[0].clientHeight);redraw();};document.getElementById('zout').onclick=()=>{zoom(1/1.6,canvases[0].clientWidth/2,canvases[0].clientHeight/2,canvases[0].clientWidth,canvases[0].clientHeight);redraw();};document.getElementById('reset').onclick=()=>{Object.assign(view,initial);redraw();};window.addEventListener('resize',redraw);redraw();
})();
</script></html>'''.replace("__DATA__", payload)
    path.write_text(markup, encoding="utf-8")


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vod-root", type=Path, required=True)
    parser.add_argument("--data-cache-root", type=Path, required=True)
    parser.add_argument("--stage2-cache-root", type=Path, required=True)
    parser.add_argument("--stage1-checkpoint", type=Path, required=True)
    parser.add_argument("--lpcr-checkpoint", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--radar-variant", default="radar_5frames")
    parser.add_argument("--split", choices=("train", "val"), default="val")
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--random-seed", type=int, default=3103)
    parser.add_argument("--occupancy-threshold", type=float, default=0.5)
    parser.add_argument("--display-limit", type=int, default=40000)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if (args.limit < 1 or args.display_limit < 1 or
            not 0 < args.occupancy_threshold < 1):
        parser.error("Limit and display limit must be positive; threshold must be in (0,1)")
    model, saved = load_stage2(args.lpcr_checkpoint, args.device, phase="lpcr")
    if (saved["stage1_sha256"] != sha256(args.stage1_checkpoint) or
            saved["radar_variant"] != args.radar_variant):
        raise ValueError("LPCR checkpoint was trained with a different Stage I")
    dataset = paired_dataset(args.vod_root, args.split, args.radar_variant,
                             include_clean=True, data_cache_root=args.data_cache_root)
    expected = cache_identity(args.stage1_checkpoint, args.vod_root,
                              args.radar_variant, float(saved["radius_m"]),
                              float(saved["gate_threshold"]), args.split,
                              [frame.frame_id for frame in dataset.frames])
    cache = Stage2FixedCache(args.stage2_cache_root / args.split, expected)
    cache.check_complete()
    indices = random.Random(args.random_seed).sample(
        range(len(dataset)), min(args.limit, len(dataset)))
    args.output_root.mkdir(parents=True, exist_ok=True)
    summary = []
    minimum = np.asarray(GRID.minimum_xyz, dtype=np.float32)
    maximum = np.asarray(GRID.maximum_xyz, dtype=np.float32)
    voxel = np.asarray(GRID.voxel_xyz, dtype=np.float32)
    for index in indices:
        row = dataset[index]
        frame_id = row["frame_id"]
        clean = row["clean_lidar"][:, :3].to(args.device)
        output, _ = cache.load(frame_id, args.device)
        conditions = model.conditions(output, float(saved["gate_threshold"]))
        if not len(conditions.hierarchy[0]):
            print(f"{frame_id}: no radar-allowed cells; skipped", flush=True)
            continue
        latent, _ = model.encode_lidar(clean, conditions)
        logits, offsets = model.lpcr(latent, conditions.hierarchy,
                                      conditions.fine_probabilities)
        scores = torch.sigmoid(logits).cpu().numpy().astype(np.float32)
        coords = conditions.hierarchy[0][:, [3, 2, 1]].cpu().numpy()
        reconstructed = (minimum + (coords.astype(np.float32) + 0.5) * voxel +
                         offsets.cpu().numpy().astype(np.float32))
        clean_xyz = clean.cpu().numpy().astype(np.float32)
        inside = ((clean_xyz >= minimum) & (clean_xyz < maximum)).all(axis=1)
        clean_roi = clean_xyz[inside]
        chosen = scores >= args.occupancy_threshold
        stem = args.output_root / f"{frame_id}_lpcr_epoch_{saved['epoch']:03d}"
        _write_ply(stem.with_name(stem.name + "_clean.ply"), clean_roi)
        _write_ply(stem.with_name(stem.name + "_reconstruction.ply"),
                   reconstructed[chosen], scores[chosen])
        np.savez_compressed(stem.with_suffix(".npz"), clean_xyz=clean_roi,
                            candidate_xyz=reconstructed, occupancy_probability=scores,
                            allowed_coordinates=coords)
        clean_display, _ = _display_subset(clean_roi, None, args.display_limit,
                                            args.random_seed + index)
        predicted_display, display_scores = _display_subset(
            reconstructed, scores, args.display_limit, args.random_seed + index + 1)
        viewer = stem.with_suffix(".html")
        _write_html(viewer, frame_id, args.split, int(saved["epoch"]), clean_display,
                    predicted_display, display_scores, args.occupancy_threshold,
                    len(clean_roi), len(reconstructed))
        stats = {"frame_id": frame_id, "clean_roi_points": len(clean_roi),
                 "allowed_cells": len(reconstructed),
                 "reconstructed_points": int(chosen.sum()),
                 "viewer": viewer.name}
        summary.append(stats)
        print(json.dumps(stats), flush=True)
    if not summary:
        raise RuntimeError("Selected frames had no radar-allowed cells")
    (args.output_root / "summary.json").write_text(json.dumps(summary, indent=2),
                                                      encoding="utf-8")
    links = "\n".join(
        f'<li><a target="viewer" href="{html.escape(item["viewer"])}">Frame {html.escape(item["frame_id"])}</a> · '
        f'{item["reconstructed_points"]:,} reconstructed points</li>' for item in summary)
    first = html.escape(summary[0]["viewer"])
    index_html = ("<!doctype html><html lang=\"en\"><meta charset=\"utf-8\">"
                  "<title>Stage II LPCR samples</title>"
                  "<style>body{margin:1rem;background:#101820;color:#edf4fa;font:16px system-ui}"
                  "a{color:#88ceff}iframe{width:100%;height:85vh;border:1px solid #526474}</style>"
                  f"<h1>Stage II LPCR {html.escape(args.split)} reconstructions</h1>"
                  "<p>Clean LiDAR is the autoencoder input. Select a frame below.</p>"
                  f"<ul>{links}</ul><iframe name=\"viewer\" src=\"{first}\" title=\"LPCR reconstruction\"></iframe>"
                  "</html>")
    (args.output_root / "index.html").write_text(index_html, encoding="utf-8")
    print(f"Open {args.output_root / 'index.html'}", flush=True)


if __name__ == "__main__":
    main()

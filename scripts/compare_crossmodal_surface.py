"""Compare raw radar-only surface proposals with optional edge-aware support."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree
import torch

from models.radar_lidar_stage1.data import VoDStage1Dataset, collate_stage1
from models.radar_lidar_stage1.radar_fov import (RADAR_FOV_SPEC, RADAR_HALF_FOV_DEG,
                                                  RADAR_MAX_RANGE_M)
from models.radar_lidar_crossmodal_v1.edge_aware_surface import edge_aware_surface
from models.radar_lidar_crossmodal_v1.train_surface_student import load_surface_student
from scripts.visualize_crossmodal_student import _write_ply


def _quality(xyz: np.ndarray, score: np.ndarray, clean: np.ndarray,
             threshold: float, tolerance: float = .2,
             selected_indices: np.ndarray | None = None) -> dict:
    selected = xyz[score >= threshold] if selected_indices is None else xyz[selected_indices]
    if not len(selected) or not len(clean):
        return {"selected_sites": len(selected),
                "precision_20cm": 0.0 if len(clean) else None,
                "clean_coverage_20cm": 0.0 if len(clean) else None}
    precision = float(np.mean(cKDTree(clean).query(selected)[0] <= tolerance))
    coverage = float(np.mean(cKDTree(selected).query(clean)[0] <= tolerance))
    return {"selected_sites": len(selected), "precision_20cm": precision,
            "clean_coverage_20cm": coverage}


def _write_html(path: Path, frame: str, epoch: int, radar: np.ndarray,
                lidar: np.ndarray, raw_xyz: np.ndarray, raw_score: np.ndarray,
                edge_xyz: np.ndarray, edge_score: np.ndarray,
                *, threshold: float, links: int) -> None:
    clouds = [{"name": "Radar", "xyz": np.round(radar, 3).tolist()},
              {"name": "Clean LiDAR", "xyz": np.round(lidar, 3).tolist()},
              {"name": "Raw predicted surfaces", "xyz": np.round(raw_xyz, 3).tolist(),
               "score": np.round(raw_score, 4).tolist()},
              {"name": "Edge-aware interpolation", "xyz": np.round(edge_xyz, 3).tolist(),
               "score": np.round(edge_score, 4).tolist()}]
    nonempty = [xyz for xyz in (radar, lidar, raw_xyz, edge_xyz) if len(xyz)]
    combined = np.concatenate(nonempty) if nonempty else np.zeros((1, 3))
    packed = json.dumps({"frame": frame, "epoch": epoch, "clouds": clouds,
                         "threshold": threshold, "links": links,
                         "minimum": combined.min(0).tolist(),
                         "maximum": combined.max(0).tolist()},
                        separators=(",", ":")).replace("<", "\\u003c")
    html = r'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Radar-only surface comparison</title><style>
body{background:#101820;color:#edf4fa;font:13px system-ui,sans-serif;margin:12px}
h1{font-size:19px;margin:0 0 5px}p{color:#b9c8d6;margin:5px 0 10px}
.controls{display:flex;gap:12px;align-items:center;flex-wrap:wrap;margin:10px 0}
input[type=range]{width:230px}button{background:#253847;color:inherit;border:1px solid #637b8b;border-radius:4px;padding:5px 9px}
.panels{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:7px}
section{min-width:0;background:#182633;border:1px solid #3b5164;padding:5px;border-radius:5px}
h2{font-size:12px;margin:0 0 5px;min-height:30px}canvas{width:100%;height:69vh;min-height:420px;background:#0c1620;touch-action:none;cursor:grab}
@media(max-width:1000px){.panels{grid-template-columns:repeat(2,minmax(0,1fr))}canvas{height:48vh;min-height:300px}}
</style><h1 id="title"></h1><p>Radar-only model output: continuous surface locations and scores. Edge-aware support uses only those predictions and learned radar features. Clean LiDAR is a reference. Every point is available; the slider filters by score, not a point cap.</p>
<p id="links"></p><div class="controls"><label>Minimum score <input type="range" id="threshold" min="0" max="1" step="0.01"><output id="value"></output></label><button id="reset">Reset view</button></div>
<div id="panels" class="panels"></div><p>Drag to rotate; Shift-drag to pan; wheel to zoom. All panels share one metric camera.</p>
<script id="data" type="application/json">__DATA__</script><script>
(()=>{const d=JSON.parse(document.getElementById('data').textContent),root=document.getElementById('panels');
document.getElementById('title').textContent=`Frame ${d.frame} · surface student epoch ${d.epoch}`;
document.getElementById('links').textContent=`Edge-aware accepted links: ${d.links.toLocaleString()}`;
const panels=d.clouds.map(cloud=>{const s=document.createElement('section'),h=document.createElement('h2'),c=document.createElement('canvas');s.append(h,c);root.append(s);return {h,c,cloud};});
const slider=document.getElementById('threshold'),value=document.getElementById('value');slider.value=d.threshold;
const center=d.minimum.map((a,i)=>(a+d.maximum[i])/2),extent=Math.hypot(...d.minimum.map((a,i)=>d.maximum[i]-a))||1;
const initial={yaw:-.65,pitch:.35,zoom:1,panX:0,panY:0},view={...initial};
function tint(score){const stops=[[40,98,190],[81,202,209],[255,225,115],[245,104,104]],q=Math.max(0,Math.min(.9999,score))*3,i=Math.floor(q),t=q-i;return `rgb(${stops[i].map((n,k)=>Math.round(n*(1-t)+stops[i+1][k]*t)).join(',')})`;}
function project(p,w,h,scale){let x=p[0]-center[0],y=p[1]-center[1],z=p[2]-center[2],cy=Math.cos(view.yaw),sy=Math.sin(view.yaw),cp=Math.cos(view.pitch),sp=Math.sin(view.pitch),u=cy*x-sy*y,depth=sy*x+cy*y;return [w/2+view.panX+u*scale,h/2+view.panY-(cp*z-sp*depth)*scale];}
function draw(){value.textContent=Number(slider.value).toFixed(2);for(let index=0;index<panels.length;index++){const {h,c,cloud}=panels[index],w=c.clientWidth,height=c.clientHeight,ratio=devicePixelRatio||1;c.width=Math.round(w*ratio);c.height=Math.round(height*ratio);const ctx=c.getContext('2d');ctx.setTransform(ratio,0,0,ratio,0,0);const scale=.88*Math.min(w,height)*view.zoom/extent;let shown=0;for(let i=0;i<cloud.xyz.length;i++){if(cloud.score&&cloud.score[i]<Number(slider.value))continue;shown++;const [x,y]=project(cloud.xyz[i],w,height,scale);if(x<0||x>w||y<0||y>height)continue;ctx.fillStyle=index===0?'#ffab40':index===1?'#45b975':tint(cloud.score[i]);let size=index<2?1.3:2;ctx.fillRect(x-size/2,y-size/2,size,size);}h.textContent=`${cloud.name}: ${shown.toLocaleString()} above score / ${cloud.xyz.length.toLocaleString()} total`;}}
let queued=false;function redraw(){if(queued)return;queued=true;requestAnimationFrame(()=>{queued=false;draw();});}
const active=new Map();for(const {c} of panels){c.addEventListener('pointerdown',e=>{c.setPointerCapture(e.pointerId);active.set(e.pointerId,[e.clientX,e.clientY]);});c.addEventListener('pointermove',e=>{let old=active.get(e.pointerId);if(!old)return;let dx=e.clientX-old[0],dy=e.clientY-old[1];active.set(e.pointerId,[e.clientX,e.clientY]);if(e.shiftKey){view.panX+=dx;view.panY+=dy;}else{view.yaw+=dx*.008;view.pitch=Math.max(-1.55,Math.min(1.55,view.pitch+dy*.008));}redraw();});for(let name of ['pointerup','pointercancel'])c.addEventListener(name,e=>active.delete(e.pointerId));c.addEventListener('wheel',e=>{e.preventDefault();view.zoom=Math.max(.1,Math.min(80,view.zoom*Math.exp(-e.deltaY*.001)));redraw();},{passive:false});}
slider.addEventListener('input',redraw);document.getElementById('reset').onclick=()=>{Object.assign(view,initial);redraw();};window.addEventListener('resize',redraw);redraw();})();
</script></html>'''.replace("__DATA__", packed)
    path.write_text(html, encoding="utf-8")


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vod-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "val", "test"), default="val")
    parser.add_argument("--frame-id", nargs="+", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--display-threshold", type=float, default=0.)
    parser.add_argument("--edge-seed-threshold", type=float, default=.25)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if not 0 <= args.display_threshold <= 1 or not 0 <= args.edge_seed_threshold <= 1:
        parser.error("Thresholds must be in [0,1]")
    student, saved = load_surface_student(args.checkpoint, args.device)
    dataset = VoDStage1Dataset(args.vod_root, args.split,
                               radar_variant=saved["radar_variant"],
                               frame_ids=args.frame_id, include_clean=True,
                               lidar_radar_fov_deg=RADAR_HALF_FOV_DEG,
                               lidar_radar_max_range_m=RADAR_MAX_RANGE_M)
    if saved.get("lidar_fov") != RADAR_FOV_SPEC:
        raise ValueError("Surface checkpoint and viewer LiDAR sector differ")
    found = {frame.frame_id for frame in dataset.frames}
    if missing := set(args.frame_id) - found:
        raise ValueError(f"Frames absent from {args.split}: {sorted(missing)}")
    args.output_root.mkdir(parents=True, exist_ok=True)
    for row in dataset:
        batch = collate_stage1([row])
        output = student(batch["radar"].to(args.device),
                         batch["radar_valid"].to(args.device))
        surface = output.surface
        raw_xyz = surface.xyz.reshape(-1, 3).cpu().numpy()
        raw_score = surface.score.reshape(-1).cpu().numpy()
        slots = surface.xyz.shape[1]
        guide = np.repeat(surface.guide.cpu().numpy(), slots, axis=0)
        groups = np.repeat(np.arange(len(surface.anchor_xyz)), slots)
        edge = edge_aware_surface(raw_xyz, raw_score, guide, groups,
                                  seed_threshold=args.edge_seed_threshold)
        edge_xyz = edge.centers_xyz()
        radar = row["radar"][:, :3].numpy()
        lidar = row["clean_lidar"][:, :3].numpy()
        stem = args.output_root / f"{row['frame_id']}_surface_epoch_{saved['epoch']:03d}"
        _write_ply(stem.with_name(stem.name + "_raw.ply"), raw_xyz, raw_score)
        _write_ply(stem.with_name(stem.name + "_edge_aware.ply"), edge_xyz, edge.scores)
        html = stem.with_suffix(".html")
        _write_html(html, row["frame_id"], saved["epoch"], radar, lidar,
                    raw_xyz, raw_score, edge_xyz, edge.scores,
                    threshold=args.display_threshold, links=edge.accepted_edges)
        report = {"frame": row["frame_id"], "checkpoint": str(args.checkpoint),
                  "epoch": saved["epoch"], "edge_seed_threshold": args.edge_seed_threshold,
                  "display_threshold": args.display_threshold,
                  "raw": _quality(raw_xyz, raw_score, lidar, args.display_threshold),
                  "edge_aware": _quality(edge_xyz, edge.scores, lidar,
                                         args.display_threshold),
                  "edge_accepted_links": edge.accepted_edges,
                  "raw_total": len(raw_xyz), "edge_total": len(edge_xyz),
                  "radar_points": len(radar), "clean_lidar_points": len(lidar)}
        selected_raw = report["raw"]["selected_sites"]
        if selected_raw:
            # Compare the same number of sites as the raw field, even if many
            # interpolated cells have identical scores at the cutoff.
            count = min(selected_raw, len(edge.scores))
            top = np.argsort(edge.scores, kind="stable")[-count:]
            report["edge_aware_matched_raw_count"] = _quality(
                edge_xyz, edge.scores, lidar, args.display_threshold,
                selected_indices=top)
            report["edge_aware_matched_raw_count"]["requested_sites"] = selected_raw
        else:
            report["edge_aware_matched_raw_count"] = None
        stem.with_suffix(".json").write_text(json.dumps(report, indent=2),
                                             encoding="utf-8")
        print(f"{row['frame_id']}: raw {len(raw_xyz)}, edge {len(edge_xyz)}, "
              f"links {edge.accepted_edges} | {html}", flush=True)


if __name__ == "__main__":
    main()

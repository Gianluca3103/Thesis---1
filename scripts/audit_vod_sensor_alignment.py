"""Audit one VoD radar/LiDAR pair and view current versus historical radar.

This reads source clouds and an optional fixed-frame cache. It does not run a
model or alter training data. Nearest-LiDAR distances are descriptive because
radar detections need not lie on the LiDAR-visible surface.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree
import torch

from Fault_Localization_Model.vod_dataset.vod_io import (
    align_radar_to_lidar, discover_vod_frames, load_vod_lidar, load_vod_radar,
    load_vod_radar_to_lidar, resolve_vod_public_root,
)
from models.radar_lidar_stage1.radar_fov import crop_lidar_to_radar_fov


def _nearest_summary(points: np.ndarray, lidar_tree: cKDTree) -> dict:
    if not len(points):
        return {"count": 0, "median_m": None, "p90_m": None,
                "within_1m_fraction": None}
    distance = lidar_tree.query(points[:, :3], k=1, workers=1)[0]
    return {"count": len(points), "median_m": float(np.median(distance)),
            "p90_m": float(np.percentile(distance, 90)),
            "within_1m_fraction": float(np.mean(distance <= 1.0))}


def _viewer(path: Path, frame_id: str, clouds: dict[str, np.ndarray]) -> None:
    packed = json.dumps({
        "frame": frame_id,
        "clouds": {name: np.round(xyz[:, :3], 3).tolist()
                   for name, xyz in clouds.items()},
    }, separators=(",", ":")).replace("<", "\\u003c")
    document = r'''<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>VoD radar/LiDAR alignment audit</title>
<style>body{background:#101820;color:#e9f2f8;font:14px system-ui;margin:16px}
label{margin-right:18px;white-space:nowrap}.panels{display:flex;gap:12px;flex-wrap:wrap}
section{flex:1 1 460px}canvas{width:100%;height:65vh;min-height:380px;background:#08131b;
border:1px solid #566c7c;touch-action:none;cursor:grab}h2{font-size:16px}
</style><h1 id="title"></h1>
<p>All clouds use the calibrated LiDAR coordinate frame. X is forward. Age 0 is
the current radar scan; older returns are accumulated and motion compensated.</p>
<div><label><input id="lidar" type="checkbox" checked>Clean LiDAR (blue)</label>
<label><input id="current" type="checkbox" checked>Current radar, age 0 (orange)</label>
<label><input id="recent" type="checkbox" checked>Radar ages −1 to −4 (yellow)</label>
<label><input id="older" type="checkbox" checked>Radar ages −5 and older (pink)</label>
<button id="reset">Reset view</button></div>
<div class="panels"><section><h2>Top view: forward X and lateral Y</h2>
<canvas id="top"></canvas></section><section><h2>Side view: forward X and height Z</h2>
<canvas id="side"></canvas></section></div>
<p>Wheel to zoom around the view center; drag to pan. Compare age 0 first, then
enable historical radar. Radar detections need not coincide exactly with LiDAR returns.</p>
<script id="data" type="application/json">__DATA__</script><script>
(()=>{const d=JSON.parse(document.getElementById('data').textContent);
document.getElementById('title').textContent=`Frame ${d.frame} · sensor alignment audit`;
const colors={lidar:'#5aabf1',current:'#ffa141',recent:'#e9e362',older:'#f46d9e'};
const boxes=[{id:'top',a:0,b:1,vertical:60},{id:'side',a:0,b:2,vertical:16}];
const views=boxes.map(spec=>({spec,canvas:document.getElementById(spec.id),
zoom:1,centerX:30,centerY:0}));
function draw(v){const c=v.canvas,w=c.clientWidth,h=c.clientHeight,r=devicePixelRatio||1;
c.width=Math.round(w*r);c.height=Math.round(h*r);const ctx=c.getContext('2d');
ctx.setTransform(r,0,0,r,0,0);ctx.clearRect(0,0,w,h);
const scale=Math.min(w/80,h/v.spec.vertical)*v.zoom;
const project=p=>[w/2+(p[v.spec.a]-v.centerX)*scale,
                   h/2-(p[v.spec.b]-v.centerY)*scale];
ctx.strokeStyle='#415363';ctx.beginPath();
for(const x of [0,10,20,30,40,50,60,70,80]){let u=project([x,0,0])[0];
ctx.moveTo(u,0);ctx.lineTo(u,h)}ctx.stroke();
for(const name of ['lidar','older','recent','current']){
if(!document.getElementById(name).checked)continue;
ctx.fillStyle=colors[name];const size=name==='lidar'?1:3;
for(const p of d.clouds[name]){const [x,y]=project(p);
if(x>=0&&x<w&&y>=0&&y<h)ctx.fillRect(x-size/2,y-size/2,size,size)}}}
for(const v of views){const c=v.canvas;let drag=null;
c.addEventListener('pointerdown',e=>{drag=[e.clientX,e.clientY];c.setPointerCapture(e.pointerId)});
c.addEventListener('pointermove',e=>{if(!drag)return;const dx=e.clientX-drag[0],dy=e.clientY-drag[1];
drag=[e.clientX,e.clientY];const scale=Math.min(c.clientWidth/80,c.clientHeight/v.spec.vertical)*v.zoom;
v.centerX-=dx/scale;v.centerY+=dy/scale;draw(v)});
for(const event of ['pointerup','pointercancel'])c.addEventListener(event,()=>drag=null);
c.addEventListener('wheel',e=>{e.preventDefault();v.zoom=Math.max(.2,Math.min(30,v.zoom*Math.exp(-e.deltaY*.001)));draw(v)},{passive:false})}
for(const name of Object.keys(colors))document.getElementById(name).addEventListener('change',()=>views.forEach(draw));
document.getElementById('reset').onclick=()=>{for(const v of views){v.zoom=1;v.centerX=30;v.centerY=0;draw(v)}};
window.addEventListener('resize',()=>views.forEach(draw));views.forEach(draw)})();
</script></html>'''.replace("__DATA__", packed)
    path.write_text(document, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vod-root", type=Path, required=True)
    parser.add_argument("--frame-id", required=True)
    parser.add_argument("--split", choices=("train", "val", "test"), default="val")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-cache-root", type=Path)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()

    saved = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    variant = saved["radar_variant"]
    frame = discover_vod_frames(args.vod_root, args.split,
                                radar_variant=variant,
                                frame_ids=[args.frame_id])[0]
    radar = load_vod_radar(frame.radar_path)
    public_root = resolve_vod_public_root(args.vod_root)
    raw_path = (public_root / "radar" /
                frame.radar_path.parent.parent.name / "velodyne" /
                f"{args.frame_id}.bin")
    raw_current = load_vod_radar(raw_path)
    age = np.rint(radar[:, 6]).astype(np.int32)
    if not np.allclose(radar[:, 6], age, atol=1e-4) or np.any(age > 0):
        raise ValueError("Radar stack has invalid scan-age values")
    current = radar[age == 0]
    current_matches_raw = (current.shape == raw_current.shape and
                           np.array_equal(current[:, :6], raw_current[:, :6]))
    lidar_from_radar = load_vod_radar_to_lidar(
        frame.lidar_calibration_path, frame.radar_calibration_path)
    aligned = align_radar_to_lidar(radar, lidar_from_radar)
    lidar = crop_lidar_to_radar_fov(load_vod_lidar(frame.lidar_path),
                                    lidar_from_radar)
    cache_checks = None
    if args.data_cache_root is not None:
        radar_cache = args.data_cache_root / args.split / "radar" / f"{args.frame_id}.pt"
        lidar_cache = args.data_cache_root / args.split / "clean_lidar" / f"{args.frame_id}.pt"
        saved_radar = torch.load(radar_cache, map_location="cpu",
                                 weights_only=True)["points"].numpy()
        saved_lidar = torch.load(lidar_cache, map_location="cpu",
                                 weights_only=True)["points"].numpy()
        cache_checks = {
            "radar_exactly_matches_calibrated_stack": bool(np.array_equal(saved_radar, aligned)),
            "lidar_exactly_matches_clean_forward_sector": bool(np.array_equal(saved_lidar, lidar)),
        }
    if not len(lidar):
        raise ValueError("The clean LiDAR forward sector is empty")
    tree = cKDTree(lidar[:, :3])
    groups = {"current_age_0": age == 0,
              "recent_ages_1_to_4": (age <= -1) & (age >= -4),
              "older_age_5_plus": age <= -5}
    nearest = {name: _nearest_summary(aligned[mask], tree)
               for name, mask in groups.items()}
    age_counts = {str(int(value)): int(np.sum(age == value))
                  for value in np.unique(age)}
    manifest_path = public_root / variant / f"doppler_manifest_{args.split}.json"
    period_s = (json.loads(manifest_path.read_text(encoding="utf-8")).get("frame_period_s")
                if manifest_path.is_file() else None)
    result = {
        "frame_id": args.frame_id, "split": args.split, "checkpoint": str(args.checkpoint),
        "checkpoint_epoch": saved.get("epoch"), "radar_variant": variant,
        "radar_path": str(frame.radar_path), "lidar_path": str(frame.lidar_path),
        "radar_to_lidar_translation_m": lidar_from_radar[:3, 3].tolist(),
        "radar_to_lidar_rotation_determinant": float(np.linalg.det(lidar_from_radar[:3, :3])),
        "current_scan_matches_released_single_radar": bool(current_matches_raw),
        "radar_age_counts": age_counts,
        "approximate_oldest_age_s": (float(-age.min() * period_s)
                                      if period_s is not None else None),
        "radar_time_note": "Scan age is ordinal; converted seconds use an approximate frame period",
        "nearest_lidar_descriptive_only": nearest,
        "fixed_cache": cache_checks,
    }
    args.output_root.mkdir(parents=True, exist_ok=True)
    report = args.output_root / f"{args.frame_id}_alignment.json"
    report.write_text(json.dumps(result, indent=2), encoding="utf-8")
    viewer = args.output_root / f"{args.frame_id}_alignment.html"
    _viewer(viewer, args.frame_id, {
        "lidar": lidar,
        "current": aligned[groups["current_age_0"]],
        "recent": aligned[groups["recent_ages_1_to_4"]],
        "older": aligned[groups["older_age_5_plus"]],
    })
    print(json.dumps(result, indent=2), flush=True)
    print(f"Open: {viewer}", flush=True)


if __name__ == "__main__":
    main()

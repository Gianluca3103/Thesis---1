"""Export exact Minkowski LiDAR reconstructions as PLY, NPZ, and offline 3D HTML."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import html
import json
import math
from pathlib import Path

import numpy as np
import torch

from models.lidar_minkowski_ae import SparseVoxelAutoencoder, _inside
from models.lidar_minkowski_ae_data import CleanVoxelFrames, summarize_counts
from models.radar_lidar_stage2.sparse_geometry import GRID
from scripts.audit_lidar_minkowski_ae import remove_coarse_blocks
from scripts.train_lidar_minkowski_ae import _counts
from scripts.visualize_stage2_lpcr import _write_ply


def voxel_centers(cells: np.ndarray) -> np.ndarray:
    """XYZ grid indices to metric voxel centers, without sampling or offsets."""
    if cells.ndim != 2 or cells.shape[1] != 3:
        raise ValueError("Expected [N,3] XYZ voxel indices")
    minimum = np.asarray(GRID.minimum_xyz, dtype=np.float32)
    size = np.asarray(GRID.voxel_xyz, dtype=np.float32)
    return minimum + (cells.astype(np.float32) + 0.5) * size


def _batched(cells: np.ndarray, device: torch.device) -> torch.Tensor:
    coords = np.column_stack((np.zeros(len(cells), np.int32), cells))
    return torch.as_tensor(coords, dtype=torch.int32, device=device)


def _prediction(model: SparseVoxelAutoencoder, input_cells: np.ndarray,
                device: torch.device, threshold: float
                ) -> tuple[np.ndarray, int]:
    latent = model.encode(_batched(input_cells, device))
    decoded = model.decode(latent, fine_targets=None, threshold=threshold)
    cutoff = math.log(threshold / (1 - threshold))
    chosen = (decoded.logits >= cutoff) & _inside(decoded.coordinates)
    cells = decoded.coordinates[chosen, 1:].cpu().numpy().astype(np.int32)
    return np.unique(cells, axis=0), latent.F.shape[0]


def _html(path: Path, *, frame_id: str, split: str, epoch: int,
          threshold: float, layers: dict[str, np.ndarray],
          clean_iou: float, masked_iou: float, missing_recall: float) -> None:
    colors = {"Clean target": "#5ac8fa", "Clean-input reconstruction": "#53db91",
              "Masked input": "#a9b4c4", "Masked-input reconstruction": "#ffa658",
              "Removed clean voxels": "#ff5d70"}
    data = {"frame": frame_id, "split": split, "epoch": epoch,
            "threshold": threshold, "clean_iou": clean_iou,
            "masked_iou": masked_iou, "missing_recall": missing_recall,
            "layers": [{"name": name, "color": colors[name], "count": len(points),
                        "points": np.round(points, 3).tolist()}
                       for name, points in layers.items()]}
    payload = json.dumps(data, separators=(",", ":")).replace("<", "\\u003c")
    markup = r'''<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Minkowski LiDAR reconstruction</title>
<style>body{margin:0;padding:14px;background:#101820;color:#eef5fa;font:14px system-ui}
a{color:#9bd3ff}h1{font-size:20px;margin:8px 0}p{color:#c4d2df}
button{background:#253847;color:inherit;border:1px solid #637b8b;border-radius:4px;padding:6px 10px;cursor:pointer}
.controls,.layers{display:flex;gap:9px 16px;flex-wrap:wrap;align-items:center;margin:11px 0}
canvas{width:100%;height:73vh;min-height:450px;background:#08131d;touch-action:none;cursor:grab;border:1px solid #3b5164}
label{white-space:nowrap}</style>
<a href="index.html">← All frames</a><h1 id="title"></h1><p id="metrics"></p>
<div class="controls"><button id="clean">Clean target</button><button id="copy">Clean → clean</button>
<button id="mask">Masked input</button><button id="masked">Masked → clean</button>
<button id="missing">Removed blocks</button><button id="reset">Reset view</button></div>
<div class="layers" id="layers"></div><canvas id="view"></canvas>
<p>All voxel centers are shown, with no point cap. Drag to rotate; Shift-drag to pan; scroll to zoom.
These are occupied voxel centers, not recovered point offsets.</p><p id="downloads"></p>
<script id="data" type="application/json">__DATA__</script>
<script>(()=>{const d=JSON.parse(document.getElementById('data').textContent);
document.getElementById('title').textContent=`${d.split} frame ${d.frame} · checkpoint epoch ${d.epoch}`;
document.getElementById('metrics').textContent=`Clean-input exact IoU ${(100*d.clean_iou).toFixed(2)}%; masked-input exact IoU ${(100*d.masked_iou).toFixed(2)}%; removed-block recall ${(100*d.missing_recall).toFixed(2)}%. Threshold ${d.threshold}.`;
const stem=`${d.split}_${d.frame}`;document.getElementById('downloads').innerHTML=`<a href="${stem}.npz">Full NPZ</a> · <a href="${stem}_target.ply">Clean PLY</a> · <a href="${stem}_reconstruction.ply">Reconstruction PLY</a> · <a href="${stem}_masked_reconstruction.ply">Masked reconstruction PLY</a>`;
const controls=document.getElementById('layers'),checks=[];
for(let i=0;i<d.layers.length;i++){const l=d.layers[i],label=document.createElement('label'),input=document.createElement('input');input.type='checkbox';input.checked=i===1;checks.push(input);label.append(input,document.createTextNode(` ${l.name} (${l.count.toLocaleString()})`));label.style.color=l.color;controls.append(label);input.onchange=redraw;}
const canvas=document.getElementById('view'),v={yaw:-.65,pitch:.35,zoom:1,panX:0,panY:0};
function project(p,w,h,s){const x=p[0]-16,y=p[1],z=p[2],cy=Math.cos(v.yaw),sy=Math.sin(v.yaw),cp=Math.cos(v.pitch),sp=Math.sin(v.pitch),u=cy*x-sy*y,q=sy*x+cy*y;return [w/2+v.panX+u*s,h/2+v.panY-(cp*z-sp*q)*s];}
let pending=false;function redraw(){if(pending)return;pending=true;requestAnimationFrame(()=>{pending=false;const ratio=devicePixelRatio||1,w=canvas.clientWidth,h=canvas.clientHeight;if(!w||!h)return;canvas.width=Math.round(w*ratio);canvas.height=Math.round(h*ratio);const ctx=canvas.getContext('2d');ctx.setTransform(ratio,0,0,ratio,0,0);ctx.clearRect(0,0,w,h);const s=.90*Math.min(w,h)*v.zoom/46;for(let i=0;i<d.layers.length;i++){if(!checks[i].checked)continue;const l=d.layers[i];ctx.fillStyle=l.color;for(const p of l.points){const [x,y]=project(p,w,h,s);if(x>=0&&x<w&&y>=0&&y<h)ctx.fillRect(x-1,y-1,2,2);}}});}
function preset(indices){checks.forEach((c,i)=>c.checked=indices.includes(i));redraw();}
document.getElementById('clean').onclick=()=>preset([0]);document.getElementById('copy').onclick=()=>preset([1]);document.getElementById('mask').onclick=()=>preset([2]);document.getElementById('masked').onclick=()=>preset([3]);document.getElementById('missing').onclick=()=>preset([3,4]);
let last=null;canvas.onpointerdown=e=>{canvas.setPointerCapture(e.pointerId);last=[e.clientX,e.clientY];};canvas.onpointermove=e=>{if(!last)return;const dx=e.clientX-last[0],dy=e.clientY-last[1];if(e.shiftKey){v.panX+=dx;v.panY+=dy;}else{v.yaw+=dx*.008;v.pitch=Math.max(-1.55,Math.min(1.55,v.pitch+dy*.008));}redraw();};for(const type of ['pointerup','pointercancel'])canvas.addEventListener(type,()=>last=null);
canvas.addEventListener('wheel',e=>{e.preventDefault();v.zoom=Math.max(.1,Math.min(80,v.zoom*Math.exp(-e.deltaY*.001)));redraw();},{passive:false});document.getElementById('reset').onclick=()=>{Object.assign(v,{yaw:-.65,pitch:.35,zoom:1,panX:0,panY:0});redraw();};window.onresize=redraw;redraw();})();</script></html>'''
    path.write_text(markup.replace("__DATA__", payload), encoding="utf-8")


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vod-root", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "val", "test"), default="val")
    parser.add_argument("--count", type=int, default=3)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--removed-block-fraction", type=float, default=0.3)
    args = parser.parse_args()
    if args.count < 1 or not 0 < args.threshold < 1 or not 0 < args.removed_block_fraction < 1:
        parser.error("Count must be positive; threshold and removed-block fraction must lie in (0,1)")
    if not torch.cuda.is_available():
        parser.error("MinkowskiEngine CUDA is required")
    saved = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if saved.get("grid") != asdict(GRID) or saved["model"]["levels"] != 1:
        raise ValueError("Expected the one-level Stage-II-grid LiDAR AE checkpoint")
    model = SparseVoxelAutoencoder(**saved["model"]).cuda().eval()
    model.load_state_dict(saved["model_state"])
    data = CleanVoxelFrames(args.vod_root, args.split, cache_root=args.cache_root)
    if args.count > len(data):
        parser.error("Count exceeds available frames")
    indices = np.linspace(0, len(data) - 1, args.count, dtype=int)
    args.output_root.mkdir(parents=True, exist_ok=True)
    summaries = []
    device = torch.device("cuda")
    for index in indices:
        row = data[int(index)]
        target = row["cells"]
        surviving, removed, _ = remove_coarse_blocks(
            target, fraction=args.removed_block_fraction, seed=3103 + int(index))
        prediction, latent_sites = _prediction(model, target, device, args.threshold)
        masked_prediction, _ = _prediction(model, surviving, device, args.threshold)
        target_t = _batched(target, device)
        clean_metrics = summarize_counts(_counts(_batched(prediction, device), target_t))
        masked_metrics = summarize_counts(_counts(_batched(masked_prediction, device), target_t))
        missing = _counts(_batched(masked_prediction, device), _batched(removed, device))
        missing_recall = missing["tp"] / missing["target"]
        stem = f"{args.split}_{row['frame_id']}"
        centers = {"Clean target": voxel_centers(target),
                   "Clean-input reconstruction": voxel_centers(prediction),
                   "Masked input": voxel_centers(surviving),
                   "Masked-input reconstruction": voxel_centers(masked_prediction),
                   "Removed clean voxels": voxel_centers(removed)}
        for label, suffix in (("Clean target", "target"),
                              ("Clean-input reconstruction", "reconstruction"),
                              ("Masked input", "masked_input"),
                              ("Masked-input reconstruction", "masked_reconstruction"),
                              ("Removed clean voxels", "removed_target")):
            _write_ply(args.output_root / f"{stem}_{suffix}.ply", centers[label])
        np.savez_compressed(args.output_root / f"{stem}.npz",
                            target_cells=target, prediction_cells=prediction,
                            masked_input_cells=surviving,
                            masked_prediction_cells=masked_prediction,
                            removed_target_cells=removed)
        _html(args.output_root / f"{stem}.html", frame_id=row["frame_id"],
              split=args.split, epoch=int(saved["epoch"]),
              threshold=args.threshold, layers=centers,
              clean_iou=clean_metrics["exact_iou"],
              masked_iou=masked_metrics["exact_iou"],
              missing_recall=missing_recall)
        summary = {"frame_id": row["frame_id"], "latent_sites": latent_sites,
                   "clean_input": clean_metrics, "masked_input": masked_metrics,
                   "removed_voxel_recall": missing_recall,
                   "html": f"{stem}.html"}
        summaries.append(summary)
        print(f"{stem}: clean IoU={clean_metrics['exact_iou']:.2%}; "
              f"masked IoU={masked_metrics['exact_iou']:.2%}; "
              f"removed recall={missing_recall:.2%}", flush=True)
    links = "".join(f'<li><a href="{html.escape(row["html"], quote=True)}">'
                    f'{html.escape(row["frame_id"])} · clean IoU '
                    f'{row["clean_input"]["exact_iou"]:.2%} · masked IoU '
                    f'{row["masked_input"]["exact_iou"]:.2%}</a></li>'
                    for row in summaries)
    index = ("<!doctype html><html lang='en'><meta charset='utf-8'>"
             "<title>Minkowski reconstruction samples</title>"
             "<style>body{background:#101820;color:#eef5fa;font:16px system-ui;margin:2rem}"
             "a{color:#9bd3ff}li{margin:1rem 0}</style>"
             "<h1>Minkowski LiDAR reconstruction samples</h1>"
             "<p>Open a frame to toggle every voxel of the clean and decoded clouds.</p>"
             f"<ul>{links}</ul></html>")
    (args.output_root / "index.html").write_text(index, encoding="utf-8")
    (args.output_root / "summary.json").write_text(
        json.dumps(summaries, indent=2), encoding="utf-8")
    print(f"Open: {args.output_root / 'index.html'}", flush=True)


if __name__ == "__main__":
    main()

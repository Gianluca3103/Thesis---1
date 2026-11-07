"""Small clean-denoiser 3D/PNG samples at the validation-selected IoU threshold.

The full validation or test split is traversed in order, preserving the same
fixed noise sequence as the published aggregate evaluation. Only a small
seeded pool of frames is retained, then low/median/high-IoU examples are shown.
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
from models.radar_lidar_stage2.clean_selected_metrics import voxel_centers
from models.radar_lidar_stage2.geometry_gate import offset_targets
from models.radar_lidar_stage2.sparse_geometry import GRID
from models.radar_lidar_stage2.sparse_residual import (
    LPCR_REVISION, VERSION, SparseResidualStage2, cosine_alpha_bars)
from models.radar_lidar_stage2.stage2_io import sha256
from scripts.train_stage2_clean_denoiser import (
    MIN_RADAR_VOXELS, RADIUS_M, REVISION, _batch, _CleanRows, _rows)
from scripts.visualize_stage2_lpcr import _write_ply


def select_examples(records: list[dict], count: int) -> list[dict]:
    """Keep examples spanning low, middle, and high exact IoU in the pool."""
    if count < 1:
        raise ValueError("Sample count must be positive")
    ordered = sorted(records, key=lambda row: (row["iou"], row["frame_id"]))
    if len(ordered) <= count:
        return ordered
    indices = [round(i * (len(ordered) - 1) / (count - 1))
               for i in range(count)] if count > 1 else [len(ordered) // 2]
    return [ordered[index] for index in indices]


def _display(points: np.ndarray, limit: int, seed: int) -> list[list[float]]:
    if len(points) > limit:
        chosen = np.sort(np.random.default_rng(seed).choice(
            len(points), limit, replace=False))
        points = points[chosen]
    return np.round(points.astype(np.float64), 3).tolist()


def _png(path: Path, *, frame_id: str, split: str, clean: np.ndarray,
         radar: np.ndarray, fine: torch.Tensor, occupied: np.ndarray,
         predicted: np.ndarray, oracle: np.ndarray, iou: float,
         oracle_iou: float) -> None:
    centers = voxel_centers(fine)
    target = centers[occupied]
    gate_xy = np.zeros((GRID.shape_zyx[1], GRID.shape_zyx[2]), dtype=np.uint8)
    local = fine[:, 1:].cpu().numpy()
    gate_xy[local[:, 1], local[:, 2]] = 1
    fig, axes = plt.subplots(2, 2, figsize=(10, 8), dpi=105,
                             constrained_layout=True)
    fig.patch.set_facecolor("#101820")
    for axis in axes.flat:
        axis.set_facecolor("#0c1620")
        axis.set_xlim(0, 32)
        axis.set_ylim(-16, 16)
        axis.set_aspect("equal")
        axis.tick_params(colors="#b9c8d6", labelsize=8)
        axis.set_xlabel("X (m)", color="#b9c8d6", fontsize=8)
        axis.set_ylabel("Y (m)", color="#b9c8d6", fontsize=8)
    axes[0, 0].set_title("Clean LiDAR context + radar", color="white")
    if len(clean):
        axes[0, 0].scatter(clean[:, 0], clean[:, 1], s=0.5, c="#aab8c6")
    if len(radar):
        axes[0, 0].scatter(radar[:, 0], radar[:, 1], s=1.2, c="#67b7ff")
    axes[0, 1].set_title("All eligible cells + occupied clean targets", color="white")
    gate_rgba = np.zeros((*gate_xy.shape, 4), dtype=np.float32)
    gate_rgba[gate_xy > 0] = (0.58, 0.36, 0.83, 0.5)
    axes[0, 1].imshow(gate_rgba, extent=(0, 32, -16, 16), origin="lower",
                      interpolation="nearest")
    if len(target):
        axes[0, 1].scatter(target[:, 0], target[:, 1], s=1.3, c="#60d796")
    for axis, chosen, name, score in (
            (axes[1, 0], oracle, "Frozen LPCR oracle", oracle_iou),
            (axes[1, 1], predicted, "One-step denoiser", iou)):
        axis.set_title(f"{name} · exact IoU {score:.1%}", color="white")
        for mask, color, size in ((occupied & ~chosen, "#ff6060", 2.1),
                                  (chosen & ~occupied, "#ffad56", 2.1),
                                  (chosen & occupied, "#60d796", 2.1)):
            points = centers[mask]
            if len(points):
                axis.scatter(points[:, 0], points[:, 1], s=size, c=color)
    fig.suptitle(f"{split} frame {frame_id} · occupancy voxel centers\n"
                 "Green: true positive  ·  Orange: false positive  ·  "
                 "Red: missed occupied cell",
                 color="white", fontsize=11)
    fig.savefig(path, facecolor=fig.get_facecolor())
    plt.close(fig)


def _html(path: Path, *, frame_id: str, split: str, epoch: int,
          threshold: float, iou: float, oracle_iou: float,
          fine: torch.Tensor, clean: np.ndarray, radar: np.ndarray,
          occupied: np.ndarray, predicted: np.ndarray,
          oracle: np.ndarray, seed: int) -> None:
    centers = voxel_centers(fine)
    layers = []
    for number, (name, color, points, limit) in enumerate((
            ("Clean LiDAR context", "#aab8c6", clean, 4000),
            ("Radar returns", "#67b7ff", radar, 2000),
            ("Eligible gate (display subset)", "#b97ae6", centers, 4000),
            ("Clean occupied targets", "#60d796", centers[occupied], 3000),
            ("Denoiser true positives", "#29e7ba",
             centers[predicted & occupied], 3000),
            ("Denoiser false positives", "#ffad56",
             centers[predicted & ~occupied], 3000),
            ("Denoiser false negatives", "#ff6060",
             centers[~predicted & occupied], 3000),
            ("Frozen LPCR oracle", "#f1e16c", centers[oracle], 3000))):
        layers.append({"name": name, "color": color, "full": len(points),
                       "points": _display(points, limit, seed + number)})
    payload = json.dumps({"frame_id": frame_id, "split": split, "epoch": epoch,
                          "threshold": threshold, "iou": iou,
                          "oracle_iou": oracle_iou,
                          "eligible_cells": len(fine),
                          "layers": layers},
                         separators=(",", ":")).replace("<", "\\u003c")
    markup = r'''<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Clean LiDAR denoiser sample</title>
<style>body{margin:0;padding:12px;background:#101820;color:#edf4fa;font:14px system-ui}
a{color:#9dcafb}button{background:#253847;color:inherit;border:1px solid #637b8b;border-radius:4px;padding:5px 9px;cursor:pointer}
.controls,.layers{display:flex;gap:8px 16px;flex-wrap:wrap;align-items:center;margin:10px 0}
label{white-space:nowrap}canvas{width:100%;height:72vh;min-height:440px;background:#0c1620;touch-action:none;cursor:grab}
p{color:#b9c8d6}h1{font-size:20px}</style>
<a href="index.html">← All samples</a><h1 id="title"></h1><p id="note"></p>
<div class="controls"><button id="clean">Clean</button><button id="gate">Gate</button>
<button id="oracle">Oracle</button><button id="denoised">Denoised errors</button>
<button id="zin">Zoom in</button><button id="zout">Zoom out</button><button id="reset">Reset</button></div>
<div id="layers" class="layers"></div><canvas id="view"></canvas>
<p>Drag to rotate · Shift-drag to pan · wheel to zoom. The gate layer displays a subset; the PNG and NPZ use every eligible voxel.</p>
<p id="downloads"></p><script id="data" type="application/json">__DATA__</script>
<script>(()=>{const d=JSON.parse(document.getElementById('data').textContent);
document.getElementById('title').textContent=`${d.split} frame ${d.frame_id} · epoch ${d.epoch} · threshold ${d.threshold.toFixed(3)}`;
document.getElementById('note').textContent=`One-step denoising of a noisy clean-LiDAR latent at t=499. Exact IoU ${Math.round(1000*d.iou)/10}%; frozen LPCR oracle ${Math.round(1000*d.oracle_iou)/10}%. ${d.eligible_cells.toLocaleString()} eligible cells. No faulty LiDAR is used.`;
const stem=`${d.split}_${d.frame_id}`;document.getElementById('downloads').innerHTML=`<a href="${stem}.png">Static PNG</a> · <a href="${stem}.npz">Full-resolution NPZ</a> · <a href="${stem}_target.ply">Clean target PLY</a> · <a href="${stem}_prediction.ply">Denoised occupancy PLY</a>`;
const controls=document.getElementById('layers'),checks=[];
for(let i=0;i<d.layers.length;i++){const l=d.layers[i],label=document.createElement('label'),input=document.createElement('input');input.type='checkbox';input.checked=[0,3,4,5,6].includes(i);checks.push(input);label.append(input,document.createTextNode(` ${l.name} (${l.full.toLocaleString()} full; ${l.points.length.toLocaleString()} shown)`));label.style.color=l.color;controls.append(label);input.addEventListener('change',redraw);}
const canvas=document.getElementById('view'),view={yaw:-.65,pitch:.35,zoom:1,panX:0,panY:0};
function project(p,w,h,s){const x=p[0]-16,y=p[1],z=p[2],cy=Math.cos(view.yaw),sy=Math.sin(view.yaw),cp=Math.cos(view.pitch),sp=Math.sin(view.pitch),u=cy*x-sy*y,v=sy*x+cy*y;return [w/2+view.panX+u*s,h/2+view.panY-(cp*z-sp*v)*s];}
let pending=false;function redraw(){if(pending)return;pending=true;requestAnimationFrame(()=>{pending=false;const ratio=devicePixelRatio||1,w=canvas.clientWidth,h=canvas.clientHeight;if(!w||!h)return;canvas.width=Math.round(w*ratio);canvas.height=Math.round(h*ratio);const ctx=canvas.getContext('2d');ctx.setTransform(ratio,0,0,ratio,0,0);ctx.clearRect(0,0,w,h);const s=.90*Math.min(w,h)*view.zoom/46;for(let i=0;i<d.layers.length;i++){if(!checks[i].checked)continue;const l=d.layers[i];ctx.fillStyle=l.color;for(const point of l.points){const [x,y]=project(point,w,h,s);if(x>=0&&x<w&&y>=0&&y<h)ctx.fillRect(x-1,y-1,2,2);}}});}
function preset(on){checks.forEach((item,i)=>item.checked=on.includes(i));redraw();}
document.getElementById('clean').onclick=()=>preset([0,1]);document.getElementById('gate').onclick=()=>preset([2,3]);document.getElementById('oracle').onclick=()=>preset([0,3,7]);document.getElementById('denoised').onclick=()=>preset([0,4,5,6]);
function zoom(k){view.zoom=Math.max(.1,Math.min(80,view.zoom*k));redraw();}let last=null;
canvas.addEventListener('pointerdown',e=>{canvas.setPointerCapture(e.pointerId);last=[e.clientX,e.clientY];});
canvas.addEventListener('pointermove',e=>{if(!last)return;const dx=e.clientX-last[0],dy=e.clientY-last[1];last=[e.clientX,e.clientY];if(e.shiftKey){view.panX+=dx;view.panY+=dy;}else{view.yaw+=dx*.008;view.pitch=Math.max(-1.55,Math.min(1.55,view.pitch+dy*.008));}redraw();});
for(const event of ['pointerup','pointercancel'])canvas.addEventListener(event,()=>last=null);
canvas.addEventListener('wheel',e=>{e.preventDefault();zoom(Math.exp(-e.deltaY*.001));},{passive:false});
document.getElementById('zin').onclick=()=>zoom(1.6);document.getElementById('zout').onclick=()=>zoom(1/1.6);document.getElementById('reset').onclick=()=>{Object.assign(view,{yaw:-.65,pitch:.35,zoom:1,panX:0,panY:0});redraw();};window.addEventListener('resize',redraw);redraw();})();</script></html>'''
    path.write_text(markup.replace("__DATA__", payload), encoding="utf-8")


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vod-root", type=Path, required=True)
    parser.add_argument("--data-cache-root", type=Path, required=True)
    parser.add_argument("--support-cache-root", type=Path, required=True)
    parser.add_argument("--lpcr-checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--sweep-json", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--split", choices=("val", "test"), required=True)
    parser.add_argument("--candidate-count", type=int, default=24)
    parser.add_argument("--count", type=int, default=6)
    parser.add_argument("--seed", type=int, default=3103)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if (args.device != "cuda" or not torch.cuda.is_available() or
            args.candidate_count < args.count or args.count < 1 or
            args.num_workers < 0):
        parser.error("CUDA, positive count, and candidate-count >= count are required")
    saved = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    identity = saved.get("identity")
    sweep = json.loads(args.sweep_json.read_text(encoding="utf-8"))
    if (not isinstance(identity, dict) or identity.get("revision") != REVISION or
            sweep.get("checkpoint_sha256") != sha256(args.checkpoint) or
            sweep.get("checkpoint_epoch") != saved.get("epoch") or
            identity.get("lpcr_sha256") != sha256(args.lpcr_checkpoint)):
        raise ValueError("Clean denoiser, LPCR, and threshold sweep do not match")
    threshold = float(sweep["selected"]["best_iou"]["threshold"])
    step_index = round(threshold * int(sweep["threshold_steps"]))
    dataset = paired_dataset(args.vod_root, args.split, identity["radar_variant"],
                             include_clean=True, data_cache_root=args.data_cache_root)
    frame_ids = [frame.frame_id for frame in dataset.frames]
    cache = RadarSupportCache(args.support_cache_root / args.split,
                              support_identity(
                                  args.split, frame_ids, identity["radar_variant"],
                                  data_cache_root=args.data_cache_root,
                                  radius_m=RADIUS_M,
                                  min_radar_voxels=MIN_RADAR_VOXELS))
    cache.check_complete()
    chosen_indices = random.Random(args.seed + (0 if args.split == "val" else 1)).sample(
        range(len(dataset)), min(args.candidate_count, len(dataset)))
    chosen_ids = {frame_ids[index] for index in chosen_indices}
    loader = DataLoader(_CleanRows(dataset, cache, None),
                        batch_size=identity["batch_size"], shuffle=False,
                        num_workers=args.num_workers, collate_fn=_rows,
                        pin_memory=True,
                        persistent_workers=args.num_workers > 0)
    saved_lpcr = torch.load(args.lpcr_checkpoint, map_location="cpu",
                            weights_only=False)
    if (saved_lpcr.get("version") != VERSION or
            saved_lpcr.get("model_revision") != LPCR_REVISION or
            saved_lpcr.get("phase") != "lpcr" or
            not saved_lpcr.get("lpcr_full_region")):
        raise ValueError("Expected the frozen full-region LPCR checkpoint")
    model = SparseResidualStage2(diffusion_target="epsilon").to(args.device)
    missing, unexpected = model.load_state_dict(saved_lpcr["model"], strict=False)
    if unexpected or any(not key.startswith("addition_head.") for key in missing):
        raise ValueError("LPCR checkpoint differs from the architecture")
    model.diffusion.load_state_dict(saved["diffusion"])
    model.eval().requires_grad_(False)
    alpha = cosine_alpha_bars(device=args.device)[int(identity["eval_timestep"])]
    generator = torch.Generator(device=args.device)
    generator.manual_seed(int(identity["seed"]))
    candidates = []
    totals = {"eligible_cells": 0, "target": 0,
              "denoised_predicted": 0, "denoised_tp": 0,
              "oracle_predicted": 0, "oracle_tp": 0}
    for rows in tqdm(loader, desc=f"Clean denoiser {args.split} sample candidates",
                     unit="batch"):
        hierarchy, clean, point_ids = _batch(rows, args.device)
        fine, coarse = hierarchy[0], hierarchy[-1]
        totals["eligible_cells"] += len(fine)
        if not len(coarse):
            for row in rows:
                if row["frame_id"] in chosen_ids:
                    candidates.append({"frame_id": row["frame_id"],
                                       "fine": torch.empty((0, 4), dtype=torch.long),
                                       "probability": np.empty(0, dtype=np.float32),
                                       "oracle_probability": np.empty(0, dtype=np.float32),
                                       "occupied": np.empty(0, dtype=bool),
                                       "predicted": np.empty(0, dtype=bool),
                                       "oracle": np.empty(0, dtype=bool),
                                       "iou": 0.0, "oracle_iou": 0.0})
            continue
        x0, _ = model.lidar_encoder(clean, coarse, point_ids)
        epsilon = torch.randn(x0.shape, dtype=x0.dtype,
                              device=args.device, generator=generator)
        xt = alpha.sqrt() * x0 + (1 - alpha).sqrt() * epsilon
        predicted_noise = model.diffusion(
            xt, torch.zeros_like(x0), torch.zeros_like(x0),
            x0.new_ones((len(x0), 1)),
            torch.zeros(len(x0), dtype=torch.bool, device=args.device),
            torch.full((len(rows),), int(identity["eval_timestep"]),
                       dtype=torch.long, device=args.device), coarse)
        recovered = (xt - (1 - alpha).sqrt() * predicted_noise) / alpha.sqrt()
        site_features = x0.new_ones((len(fine), 1))
        logits, _ = model.lpcr(recovered.clamp(-1, 1), hierarchy, site_features)
        oracle_logits, _ = model.lpcr(x0, hierarchy, site_features)
        target, _, _ = offset_targets(clean, fine, point_ids)
        predicted_all = (torch.floor(logits.sigmoid() * 1000)
                         .long().clamp(0, 1000) >= step_index)
        oracle_all = (torch.floor(oracle_logits.sigmoid() * 1000)
                      .long().clamp(0, 1000) >= step_index)
        totals["target"] += int(target.sum())
        totals["denoised_predicted"] += int(predicted_all.sum())
        totals["denoised_tp"] += int((predicted_all & target).sum())
        totals["oracle_predicted"] += int(oracle_all.sum())
        totals["oracle_tp"] += int((oracle_all & target).sum())
        if not any(row["frame_id"] in chosen_ids for row in rows):
            continue
        probability = logits.sigmoid()
        oracle_probability = oracle_logits.sigmoid()
        for frame, row in enumerate(rows):
            if row["frame_id"] not in chosen_ids:
                continue
            frame_mask = fine[:, 0] == frame
            local = fine[frame_mask].cpu().clone()
            local[:, 0] = 0
            probabilities = probability[frame_mask].cpu()
            oracle_probabilities = oracle_probability[frame_mask].cpu()
            occupied = target[frame_mask].cpu().numpy()
            predicted = (torch.floor(probabilities * 1000).long().clamp(0, 1000)
                         >= step_index).numpy()
            oracle = (torch.floor(oracle_probabilities * 1000).long().clamp(0, 1000)
                      >= step_index).numpy()
            tp = int(np.count_nonzero(predicted & occupied))
            oracle_tp = int(np.count_nonzero(oracle & occupied))
            candidates.append({"frame_id": row["frame_id"], "fine": local,
                               "probability": probabilities.numpy(),
                               "oracle_probability": oracle_probabilities.numpy(),
                               "occupied": occupied, "predicted": predicted,
                               "oracle": oracle,
                               "iou": tp / max(int(np.count_nonzero(
                                   predicted | occupied)), 1),
                               "oracle_iou": oracle_tp / max(int(np.count_nonzero(
                                   oracle | occupied)), 1)})
    if len(candidates) != len(chosen_ids):
        raise RuntimeError(f"Captured {len(candidates)} of {len(chosen_ids)} frames")
    def micro_iou(tp: int, predicted: int) -> float:
        return tp / max(predicted + totals["target"] - tp, 1)
    aggregate = {
        **totals, "all_eligible_positive_iou": totals["target"] /
        max(totals["eligible_cells"], 1),
        "denoised_iou": micro_iou(totals["denoised_tp"],
                                   totals["denoised_predicted"]),
        "frozen_lpcr_oracle_iou": micro_iou(totals["oracle_tp"],
                                             totals["oracle_predicted"])}
    print(f"Full {args.split}: gate-all-positive IoU="
          f"{aggregate['all_eligible_positive_iou']:.2%}, "
          f"denoised IoU={aggregate['denoised_iou']:.2%}, "
          f"frozen LPCR oracle IoU={aggregate['frozen_lpcr_oracle_iou']:.2%} "
          f"at threshold {threshold:.3f}", flush=True)
    examples = select_examples(candidates, args.count)
    args.output_root.mkdir(parents=True, exist_ok=True)
    indices = {frame.frame_id: index for index, frame in enumerate(dataset.frames)}
    summary = []
    minimum = np.asarray(GRID.minimum_xyz)
    maximum = np.asarray(GRID.maximum_xyz)
    for rank, item in enumerate(examples):
        frame_id = item["frame_id"]
        source = dataset[indices[frame_id]]
        clean = source["clean_lidar"][:, :3].numpy()
        radar = source["radar"][:, :3].numpy()
        clean = clean[((clean >= minimum) & (clean < maximum)).all(1)]
        radar = radar[((radar >= minimum) & (radar < maximum)).all(1)]
        fine = item["fine"]
        centers = voxel_centers(fine)
        stem = f"{args.split}_{frame_id}"
        np.savez_compressed(
            args.output_root / f"{stem}.npz", clean_xyz=clean, radar_xyz=radar,
            fine_bzyx=fine.numpy(), occupancy_probability=item["probability"],
            oracle_probability=item["oracle_probability"],
            clean_occupied=item["occupied"], threshold=threshold,
            checkpoint_epoch=int(saved["epoch"]), eval_timestep=int(identity["eval_timestep"]))
        _write_ply(args.output_root / f"{stem}_target.ply",
                   centers[item["occupied"]].astype(np.float32))
        _write_ply(args.output_root / f"{stem}_prediction.ply",
                   centers[item["predicted"]].astype(np.float32),
                   item["probability"][item["predicted"]].astype(np.float32))
        _png(args.output_root / f"{stem}.png", frame_id=frame_id,
             split=args.split, clean=clean, radar=radar, fine=fine,
             occupied=item["occupied"], predicted=item["predicted"],
             oracle=item["oracle"], iou=item["iou"],
             oracle_iou=item["oracle_iou"])
        _html(args.output_root / f"{stem}.html", frame_id=frame_id,
              split=args.split, epoch=int(saved["epoch"]), threshold=threshold,
              iou=item["iou"], oracle_iou=item["oracle_iou"],
              fine=fine, clean=clean, radar=radar,
              occupied=item["occupied"], predicted=item["predicted"],
              oracle=item["oracle"], seed=args.seed + rank * 10)
        row = {"frame_id": frame_id, "split": args.split, "rank": rank,
               "iou": item["iou"], "oracle_iou": item["oracle_iou"],
               "eligible_cells": len(fine),
               "clean_target_cells": int(item["occupied"].sum()),
               "predicted_cells": int(item["predicted"].sum()),
               "png": f"{stem}.png", "html": f"{stem}.html"}
        summary.append(row)
        print(f"{args.split} {frame_id}: IoU={row['iou']:.2%}, "
              f"oracle={row['oracle_iou']:.2%}, "
              f"targets={row['clean_target_cells']:,}, "
              f"predicted={row['predicted_cells']:,}", flush=True)
    (args.output_root / "summary.json").write_text(
        json.dumps({"checkpoint": str(args.checkpoint), "epoch": saved["epoch"],
                    "threshold": threshold, "candidate_count": len(candidates),
                    "selection": "IoU quantiles within a seeded candidate pool",
                    "aggregate": aggregate, "samples": summary}, indent=2)
        + "\n", encoding="utf-8")
    cards = "\n".join(
        f'<li><a href="{html.escape(row["html"], quote=True)}">'
        f'{html.escape(row["split"])} {html.escape(row["frame_id"])} · '
        f'IoU {row["iou"]:.1%} · oracle {row["oracle_iou"]:.1%}</a>'
        f' · <a href="{html.escape(row["png"], quote=True)}">PNG</a></li>'
        for row in summary)
    (args.output_root / "index.html").write_text(
        "<!doctype html><html lang='en'><meta charset='utf-8'>"
        "<title>Clean denoiser samples</title><style>body{margin:30px auto;"
        "max-width:850px;padding:0 18px;background:#101820;color:#eef5fa;"
        "font:16px system-ui}a{color:#9dcafb}li{margin:13px 0}</style>"
        f"<h1>Clean denoiser · {html.escape(args.split)}</h1>"
        f"<p>{len(summary)} examples spanning low to high IoU among "
        f"{len(candidates)} seeded candidates. Click a frame for the 3D viewer "
        "or its static PNG.</p><ul>" + cards + "</ul></html>",
        encoding="utf-8")
    print(f"Open: {args.output_root / 'index.html'}", flush=True)


if __name__ == "__main__":
    main()

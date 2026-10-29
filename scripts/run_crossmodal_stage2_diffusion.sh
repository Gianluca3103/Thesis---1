#!/usr/bin/env bash
set -Eeuo pipefail

# Foreground Stage-II trial using the trained cross-modal radar-only student.
# Reuse the LiDAR-only LPCR and severe-fog cache; rebuild the Stage-I gate.
REPO=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)
BASE=${BASE:-/mnt/3D10B36523559581/Gianluca}
VOD=${VOD:-/mnt/3D10B36523559581/View-of-Delft/view_of_delft_detection_PUBLIC/view_of_delft_PUBLIC}
PY=${PY:-$BASE/Sensor-Fusion/.venv_model_v2/bin/python}
DATA=${DATA:-/media/arrubuntu20/EA0A117F0A114A49/sensor_fusion_outputs}
PREV=${PREV:-$DATA/radar_stage2_ray0_20261028}
STAGE1_CKPT=${STAGE1_CKPT:-$DATA/crossmodal_fov70_refined_noaug_30ep_20261015/full/student/best_radar_iou.ckpt}
LPCR_CKPT=${LPCR_CKPT:-$PREV/stage2_sparse_lpcr_fullregion_v9/best.ckpt}
FOG_CACHE=${FOG_CACHE:-$PREV/stage2_severe_fog_s5_v1}
OUT=${OUT:-$DATA/crossmodal_stage2_ray025_20261029}
WORKERS=${WORKERS:-8}
BATCH_SIZE=${BATCH_SIZE:-16}
STAGE2_CACHE=$OUT/stage2_gate
DIFF=$OUT/diffusion

fail() { printf '%s\n' "$*" >&2; exit 1; }
on_error() {
  local status=$?
  printf 'Cross-modal Stage II stopped at line %s (exit %s). Outputs remain in %s\n' \
    "$1" "$status" "$OUT" >&2
  exit "$status"
}
trap 'on_error $LINENO' ERR

[[ -x "$PY" && -s "$STAGE1_CKPT" && -s "$LPCR_CKPT" ]] ||
  fail "Python, cross-modal student, or LPCR checkpoint is missing"
[[ -s "$FOG_CACHE/train/manifest.json" && -s "$FOG_CACHE/val/manifest.json" ]] ||
  fail "Full train and validation severe-fog cache manifests are required"
[[ "$WORKERS" =~ ^[0-9]+$ && "$BATCH_SIZE" =~ ^[1-9][0-9]*$ ]] ||
  fail "WORKERS must be nonnegative and BATCH_SIZE must be positive"
command -v flock >/dev/null || fail "flock is required"

mkdir -p "$OUT" "$OUT/matplotlib" "$STAGE2_CACHE"
exec 9>"$OUT/run.lock"
flock -n 9 || fail "Another process is using $OUT"
export PYTHONPATH="$REPO"
export LD_LIBRARY_PATH="$BASE/stage2-openblas-runtime/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export OMP_NUM_THREADS=8 MPLCONFIGDIR="$OUT/matplotlib" MPLBACKEND=Agg
cd "$REPO"

"$PY" -c '
import sys
from pathlib import Path
import models.radar_lidar_stage2.stage2_io as stage2_io
expected = Path(sys.argv[1]).resolve()
loaded = Path(stage2_io.__file__).resolve()
if not loaded.is_relative_to(expected):
    raise RuntimeError(f"Loaded Stage-II code from {loaded}, expected {expected}")
print(f"Stage-II code: {loaded}", flush=True)
' "$REPO"

"$PY" -c '
import sys, torch
from models.radar_lidar_stage1.radar_fov import RADAR_FOV_SPEC
s = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
assert s.get("version") == "crossmodal_student_v5_multiscale_refined_raw64"
assert s.get("lidar_fov") == RADAR_FOV_SPEC
assert not s["config"].get("decoder_from_conditioned", False)
assert s.get("radar_variant")
assert s.get("selected_threshold") is not None
print("Stage I: epoch", s.get("epoch"), "radar", s["radar_variant"],
      "selected gate", s["selected_threshold"], flush=True)
' "$STAGE1_CKPT"
RADAR_VARIANT=$("$PY" -c 'import sys,torch; print(torch.load(sys.argv[1],map_location="cpu",weights_only=False)["radar_variant"])' "$STAGE1_CKPT")
GATE=$("$PY" -c 'import sys,torch; print(torch.load(sys.argv[1],map_location="cpu",weights_only=False)["selected_threshold"])' "$STAGE1_CKPT")
[[ "$GATE" != "None" ]] || fail "Stage-I checkpoint lacks a selected threshold"
printf 'Output: %s\nRadar variant: %s\nStage-I gate: %s\n' "$OUT" "$RADAR_VARIANT" "$GATE"
df -h "$OUT"

# Reuse an aligned 20-frame cache only when the current loader validates its
# manifest, source hash, FOV and complete train/validation frame lists.
RAW=
for candidate in \
  "$OUT/fixed_frames" \
  "$DATA/crossmodal_fov70_refined_noaug_30ep_20261015/fixed_frames" \
  "$DATA/crossmodal_noaug_30ep_20261015/fixed_frames"; do
  if [[ -s "$candidate/train/manifest.json" && -s "$candidate/val/manifest.json" ]] &&
     "$PY" -c '
import sys
from pathlib import Path
from models.radar_lidar_crossmodal_v1.pipeline_data import paired_dataset
for split in ("train", "val"):
    paired_dataset(Path(sys.argv[1]), split, sys.argv[2],
                   data_cache_root=Path(sys.argv[3]))
' "$VOD" "$RADAR_VARIANT" "$candidate" >/dev/null 2>&1; then
    RAW=$candidate
    break
  fi
done
if [[ -z "$RAW" ]]; then
  RAW=$OUT/fixed_frames
  printf '\n=== Build aligned 20-frame train and validation cache ===\n'
  for split in train val; do
    "$PY" -u -m scripts.cache_crossmodal_fixed_frames \
      --vod-root "$VOD" --radar-variant "$RADAR_VARIANT" \
      --cache-root "$RAW" --split "$split" --num-workers "$WORKERS"
  done
else
  printf 'Reusing validated aligned frame cache: %s\n' "$RAW"
fi

printf '\n=== Check disk space from a 16-frame gate-cache pilot ===\n'
PILOT=$OUT/stage2_gate_pilot
"$PY" -u -m scripts.cache_stage2_sparse \
  --vod-root "$VOD" --data-cache-root "$RAW" \
  --stage1-checkpoint "$STAGE1_CKPT" --cache-root "$PILOT" \
  --radar-variant "$RADAR_VARIANT" --gate-threshold "$GATE" \
  --split train --limit 16 --num-workers "$WORKERS" --device cuda
"$PY" -c '
import shutil, sys
from pathlib import Path
from models.radar_lidar_crossmodal_v1.pipeline_data import paired_dataset
pilot, target, vod = map(Path, sys.argv[1:4])
variant = sys.argv[4]
sizes = [p.stat().st_size for p in (pilot / "train").glob("*.npz")]
if len(sizes) != 16:
    raise SystemExit("Gate pilot did not write 16 frames")
count = sum(len(paired_dataset(vod, split, variant, id_only=True))
            for split in ("train", "val"))
already = sum(len(list((target / split).glob("*.npz")))
              for split in ("train", "val"))
estimate = int(sum(sizes) / len(sizes) * max(count - already, 0) * 1.3)
free = shutil.disk_usage(target).free
reserve = 10 * 1024**3
print(f"Estimated remaining gate cache: {estimate / 1024**3:.1f} GiB; "
      f"free: {free / 1024**3:.1f} GiB", flush=True)
if free < estimate + reserve:
    raise SystemExit("Insufficient space for gate cache plus 10 GiB reserve")
' "$PILOT" "$STAGE2_CACHE" "$VOD" "$RADAR_VARIANT"

printf '\n=== Build cross-modal Stage-I gate cache ===\n'
"$PY" -u -m scripts.cache_stage2_sparse \
  --vod-root "$VOD" --data-cache-root "$RAW" \
  --stage1-checkpoint "$STAGE1_CKPT" --cache-root "$STAGE2_CACHE" \
  --radar-variant "$RADAR_VARIANT" --gate-threshold "$GATE" \
  --split both --num-workers "$WORKERS" --device cuda

printf '\n=== Verify cached radar conditioning comes from the cross-modal student ===\n'
"$PY" -c '
import json, sys, torch
from pathlib import Path
from models.radar_lidar_crossmodal_v1.pipeline_data import paired_dataset
from models.radar_lidar_stage2.stage2_cache import Stage2FixedCache
from models.radar_lidar_stage2.stage2_io import load_stage1, sha256, stage1_frame

ckpt, cache_root, vod, raw = map(Path, sys.argv[1:5])
variant = sys.argv[5]
torch.set_grad_enabled(False)
manifest = json.loads((cache_root / "val" / "manifest.json").read_text())
if manifest["stage1_sha256"] != sha256(ckpt):
    raise ValueError("Gate cache belongs to another Stage-I checkpoint")
model, saved = load_stage1(ckpt, "cuda", variant)
row = paired_dataset(vod, "val", variant, data_cache_root=raw, limit=1)[0]
live = stage1_frame(model, row["radar"].cuda(), float(saved.get("radius_m", 3.2)))
cached, _ = Stage2FixedCache(cache_root / "val", manifest).load(row["frame_id"], "cpu")
if not torch.equal(cached["conditioning_coords"], live["conditioning_coords"].cpu()):
    raise ValueError("Cached conditioning coordinates differ from the cross-modal student")
torch.testing.assert_close(cached["conditioning_features"],
                           live["conditioning_features"].cpu(), rtol=1e-5,
                           atol=1e-6)
sites = len(cached["conditioning_coords"])
digest = manifest["stage1_sha256"]
print(f"Verified {sites} stride-8 conditioning sites from Stage-I SHA256 "
      f"{digest[:12]}", flush=True)
' "$STAGE1_CKPT" "$STAGE2_CACHE" "$VOD" "$RAW" "$RADAR_VARIANT"

printf '\n=== Train Stage-II diffusion with the same ray loss as ray_on ===\n'
resume=()
if [[ -s "$DIFF/last.ckpt" ]]; then
  resume=(--resume "$DIFF/last.ckpt")
elif [[ -e "$DIFF/metrics.jsonl" || -e "$DIFF/best.ckpt" ]]; then
  fail "Incomplete diffusion state in $DIFF; inspect before resuming"
fi
"$PY" -u -m scripts.train_stage2_sparse \
  --phase diffusion --vod-root "$VOD" --data-cache-root "$RAW" \
  --stage2-cache-root "$STAGE2_CACHE" --fault-cache-root "$FOG_CACHE" \
  --stage1-checkpoint "$STAGE1_CKPT" --lpcr-checkpoint "$LPCR_CKPT" \
  --output-root "$DIFF" --radar-variant "$RADAR_VARIANT" \
  --gate-threshold "$GATE" --epochs 50 --lr 0.0002 \
  --batch-size "$BATCH_SIZE" --num-workers "$WORKERS" \
  --residual-weight 1 --reconstruction-weight 1 \
  --addition-weight 1 --healthy-weight 1 \
  --ray-hit-weight 0.25 --ray-extra-weight 0.25 \
  --device cuda "${resume[@]}" 2>&1 | tee -a "$OUT/training.log"
[[ -s "$DIFF/best.ckpt" ]] || fail "Diffusion produced no best checkpoint"
printf 'Finished: %s\n' "$DIFF/best.ckpt"

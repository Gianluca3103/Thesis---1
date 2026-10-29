#!/usr/bin/env bash
set -Eeuo pipefail

# Fresh cross-modal encoders, teacher, student, LiDAR decoder and Stage II.
# Only aligned raw radar/clean-LiDAR frames and the explicitly selected revised
# fog cache are reused. The run stays in the foreground and resumes on rerun.
REPO=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)
BASE=${BASE:-/mnt/3D10B36523559581/Gianluca}
VOD=${VOD:-/mnt/3D10B36523559581/View-of-Delft/view_of_delft_detection_PUBLIC/view_of_delft_PUBLIC}
PY=${PY:-$BASE/Sensor-Fusion/.venv_model_v2/bin/python}
DATA=${DATA:-/media/arrubuntu20/EA0A117F0A114A49/sensor_fusion_outputs}
OUT=${OUT:-$DATA/crossmodal_full_refog_20261029}
RAW=${RAW:-$DATA/crossmodal_fov70_refined_noaug_30ep_20261015/fixed_frames}
FOG_CACHE=${FOG_CACHE:?Set FOG_CACHE to the revised train/val fog-cache root}
WORKERS=${WORKERS:-8}
BATCH_SIZE=${BATCH_SIZE:-16}

fail() { printf '%s\n' "$*" >&2; exit 1; }
on_error() {
  local status=$?
  printf 'Full retraining stopped at line %s (exit %s). Outputs remain in %s\n' \
    "$1" "$status" "$OUT" >&2
  exit "$status"
}
trap 'on_error $LINENO' ERR

[[ -x "$PY" && -s "$RAW/train/manifest.json" && -s "$RAW/val/manifest.json" ]] ||
  fail "Python or validated aligned-frame cache is missing"
[[ -s "$FOG_CACHE/train/manifest.json" && -s "$FOG_CACHE/val/manifest.json" ]] ||
  fail "Revised fog cache needs train and val manifests: $FOG_CACHE"
[[ "$WORKERS" =~ ^[1-9][0-9]*$ && "$BATCH_SIZE" =~ ^[1-9][0-9]*$ ]] ||
  fail "WORKERS and BATCH_SIZE must be positive"
command -v flock >/dev/null || fail "flock is required"

mkdir -p "$OUT" "$OUT/matplotlib"
exec 9>"$OUT/run.lock"
flock -n 9 || fail "Another process is using $OUT"
export PYTHONPATH="$REPO"
export LD_LIBRARY_PATH="$BASE/stage2-openblas-runtime/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export OMP_NUM_THREADS=8 MPLCONFIGDIR="$OUT/matplotlib" MPLBACKEND=Agg
cd "$REPO"

"$PY" -c '
import json, sys
from pathlib import Path
from models.radar_lidar_crossmodal_v1.pipeline_data import paired_dataset
from models.radar_lidar_stage2.fault_data import SevereFogCache

vod, raw, fog = map(Path, sys.argv[1:4])
variant = "radar_20frames_verified_doppler_radial"
for split in ("train", "val"):
    frames = paired_dataset(vod, split, variant, data_cache_root=raw)
    cache = SevereFogCache(fog, split, [frame.frame_id for frame in frames.frames])
    for frame in (frames.frames[0], frames.frames[-1]):
        cache.load(frame.frame_id)
    manifest = json.loads((fog / split / "manifest.json").read_text())
    sim_hash = manifest.get("fog_simulator_sha256", "?")
    print(f"{split}: {len(frames)} aligned frames and revised fog entries; "
          f"simulator SHA256 {sim_hash[:12]}", flush=True)
' "$VOD" "$RAW" "$FOG_CACHE"
df -h "$OUT"

printf '\n=== Retrain encoders, cross-attention teacher and radar-only student ===\n'
OUT="$OUT/crossmodal" RAW="$RAW" STUDENT_CACHE_FORMAT=compact \
  bash "$REPO/scripts/train_crossmodal_noaug_30ep.sh"
STAGE1_CKPT=$OUT/crossmodal/full/student/best_radar_iou.ckpt
[[ -s "$STAGE1_CKPT" ]] || fail "Full cross-modal student checkpoint missing"

printf '\n=== Retrain LiDAR-only LPCR from scratch ===\n'
LPCR=$OUT/lpcr
mkdir -p "$LPCR"
lpcr_resume=()
if [[ -s "$LPCR/last.ckpt" ]]; then
  lpcr_resume=(--resume "$LPCR/last.ckpt")
elif [[ -e "$LPCR/metrics.jsonl" || -e "$LPCR/best.ckpt" ]]; then
  fail "Incomplete LPCR state in $LPCR; inspect before resuming"
fi
"$PY" -u -m scripts.train_stage2_sparse \
  --phase lpcr --vod-root "$VOD" --data-cache-root "$RAW" \
  --output-root "$LPCR" --radar-variant radar_20frames_verified_doppler_radial \
  --epochs 30 --lr 0.001 --batch-size 8 --num-workers "$WORKERS" \
  --paper-metrics-every 5 --paper-metrics-frames 4 \
  --device cuda "${lpcr_resume[@]}" 2>&1 | tee -a "$LPCR/console.log"
[[ -s "$LPCR/best.ckpt" ]] || fail "LPCR produced no best checkpoint"

printf '\n=== Fresh Stage-II gate cache and diffusion on revised fog ===\n'
STAGE1_CKPT="$STAGE1_CKPT" LPCR_CKPT="$LPCR/best.ckpt" \
  FOG_CACHE="$FOG_CACHE" RAW="$RAW" OUT="$OUT/stage2" \
  BATCH_SIZE="$BATCH_SIZE" WORKERS="$WORKERS" \
  bash "$REPO/scripts/run_crossmodal_stage2_diffusion.sh"

printf '\nFull retraining complete.\nStage I: %s\nLPCR: %s\nDiffusion: %s\n' \
  "$STAGE1_CKPT" "$LPCR/best.ckpt" "$OUT/stage2/diffusion/best.ckpt"

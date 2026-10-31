#!/usr/bin/env bash
# Matched raw/distilled/teacher Stage-I heads and held-out validation comparison.
set -euo pipefail

REPO=${REPO:-/mnt/3D10B36523559581/Gianluca/sensor-fusion-fault-localization}
BASE=${BASE:-/mnt/3D10B36523559581/Gianluca}
DATA=${DATA:-/media/arrubuntu20/EA0A117F0A114A49/sensor_fusion_outputs}
VOD=${VOD:-/mnt/3D10B36523559581/View-of-Delft/view_of_delft_detection_PUBLIC/view_of_delft_PUBLIC}
PY=${PY:-$BASE/Sensor-Fusion/.venv_model_v2/bin/python}
RAW=${RAW:-$DATA/crossmodal_fov70_refined_noaug_30ep_20261015/fixed_frames}
SOURCE_STUDENT=${SOURCE_STUDENT:-$DATA/crossmodal_full_refog_20261029/crossmodal/full/student/best_radar_iou.ckpt}
OUT=${OUT:-$DATA/fine_distillation_control_1p6m64_20261031}
EPOCHS=${EPOCHS:-25}
BATCH_SIZE=${BATCH_SIZE:-8}
WORKERS=${WORKERS:-8}

for required in "$PY" "$SOURCE_STUDENT" \
                "$RAW/train/manifest.json" "$RAW/val/manifest.json" \
                "$REPO/configs/crossmodal_fine_association_1p6m64.json"; do
  [[ -s "$required" ]] || { printf 'Missing required file: %s\n' "$required" >&2; exit 1; }
done
cd "$REPO"
export PYTHONPATH="$REPO" OMP_NUM_THREADS=8 TMPDIR="$DATA"
export LD_LIBRARY_PATH="$BASE/stage2-openblas-runtime/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

smoke="$OUT/smoke"
if [[ ! -s "$smoke/last.ckpt" ]]; then
  "$PY" -u -m scripts.train_fine_distillation_control \
    --vod-root "$VOD" --data-cache-root "$RAW" \
    --source-student "$SOURCE_STUDENT" --output-root "$smoke" \
    --epochs 1 --train-limit "$BATCH_SIZE" --val-limit 8 \
    --quick-val-frames 8 --batch-size "$BATCH_SIZE" \
    --num-workers 2 --device cuda
fi
"$PY" -u -m scripts.compare_fine_distillation_control \
  --vod-root "$VOD" --data-cache-root "$RAW" \
  --checkpoint "$smoke/last.ckpt" --output-root "$smoke/comparison" \
  --limit 8 --num-workers 2 --viewer-limit 1 --device cuda

train="$OUT/train"
resume=()
if [[ -s "$train/last.ckpt" ]]; then
  resume=(--resume "$train/last.ckpt")
elif [[ -e "$train/metrics.jsonl" ]]; then
  printf 'Incomplete training state in %s\n' "$train" >&2
  exit 1
fi
"$PY" -u -m scripts.train_fine_distillation_control \
  --vod-root "$VOD" --data-cache-root "$RAW" \
  --source-student "$SOURCE_STUDENT" --output-root "$train" \
  --epochs "$EPOCHS" --batch-size "$BATCH_SIZE" \
  --num-workers "$WORKERS" --device cuda "${resume[@]}"

"$PY" -u -m scripts.compare_fine_distillation_control \
  --vod-root "$VOD" --data-cache-root "$RAW" \
  --checkpoint "$train/last.ckpt" --output-root "$OUT/comparison_val" \
  --limit 1296 --num-workers 2 --viewer-limit 5 \
  --top-k 64 --display-threshold 0.6 --device cuda

printf 'Comparison: %s\nViewer: %s\n' \
  "$OUT/comparison_val/comparison_summary.json" \
  "$OUT/comparison_val/index.html"

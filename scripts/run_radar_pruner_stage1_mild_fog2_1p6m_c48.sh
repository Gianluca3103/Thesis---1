#!/usr/bin/env bash
# Compare 48 base channels against the 16-channel 1.6 m pruner.
set -euo pipefail

REPO=${REPO:-/media/arrubuntu20/EA0A117F0A114A49/Thesis-1-hierarchical-stage1-20261101}
BASE=${BASE:-/mnt/3D10B36523559581/Gianluca}
DATA=${DATA:-/media/arrubuntu20/EA0A117F0A114A49/sensor_fusion_outputs}
PY=${PY:-$BASE/Sensor-Fusion/.venv_model_v2/bin/python}
VOD=${VOD:-/mnt/3D10B36523559581/View-of-Delft/view_of_delft_detection_PUBLIC/view_of_delft_PUBLIC}
RAW=${RAW:-$DATA/crossmodal_fov70_refined_noaug_30ep_20261015/fixed_frames}
MILD=${MILD:-$DATA/stage2_mild_fog2_20frame_20261101}
SUPPORT=${SUPPORT:-$DATA/radar_pruner3vox_1p6m_mild_fog2_20261103/support_masks}
OUT=${OUT:-$DATA/radar_pruner3vox_1p6m_c48_mild_fog2_20261103}

for required in "$PY" "$MILD/train/manifest.json" "$MILD/val/manifest.json" \
  "$RAW/train/manifest.json" "$RAW/val/manifest.json" \
  "$SUPPORT/train/manifest.json" "$SUPPORT/val/manifest.json" \
  "$REPO/scripts/train_radar_pruner_stage1.py" \
  "$REPO/scripts/evaluate_radar_pruner_stage1.py"; do
  [[ -s "$required" ]] || { printf 'Missing required file: %s\n' "$required" >&2; exit 1; }
done

mkdir -p "$OUT"
cd "$REPO"
export PYTHONPATH="$REPO" OMP_NUM_THREADS=8
export LD_LIBRARY_PATH="$BASE/stage2-openblas-runtime/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

resume_args=()
if [[ -s "$OUT/last.ckpt" ]]; then
  resume_args=(--resume "$OUT/last.ckpt")
  printf 'Resuming 48-channel run from %s\n' "$OUT/last.ckpt"
fi

printf 'Training 48-channel pruner inside the cached 1.6 m domain.\n'
"$PY" -u -m scripts.train_radar_pruner_stage1 \
  --vod-root "$VOD" --data-cache-root "$RAW" \
  --faulty-root "$MILD" --support-cache-root "$SUPPORT" \
  --output-root "$OUT" --radius-m 1.6 --channels 48 \
  --target-eligible-recall 0.90 --epochs 25 --val-limit 64 \
  --num-workers 2 --device cuda "${resume_args[@]}" \
  2>&1 | tee "$OUT/train.log"

printf 'Evaluating on 256 calibration and 256 held-out frames.\n'
"$PY" -u -m scripts.evaluate_radar_pruner_stage1 \
  --vod-root "$VOD" --data-cache-root "$RAW" \
  --faulty-root "$MILD" --support-cache-root "$SUPPORT" \
  --checkpoint "$OUT/best.ckpt" \
  --output "$OUT/heldout_3vox_1p6m_c48.json" \
  --limit 512 --device cuda \
  2>&1 | tee "$OUT/evaluate.log"

printf 'Completed: %s\n' "$OUT/heldout_3vox_1p6m_c48.json"

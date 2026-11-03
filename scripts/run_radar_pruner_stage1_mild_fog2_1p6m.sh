#!/usr/bin/env bash
# Three distinct radar voxels within 1.6 m; select for 90% eligible recall.
set -euo pipefail

REPO=${REPO:-/media/arrubuntu20/EA0A117F0A114A49/Thesis-1-hierarchical-stage1-20261101}
BASE=${BASE:-/mnt/3D10B36523559581/Gianluca}
DATA=${DATA:-/media/arrubuntu20/EA0A117F0A114A49/sensor_fusion_outputs}
PY=${PY:-$BASE/Sensor-Fusion/.venv_model_v2/bin/python}
VOD=${VOD:-/mnt/3D10B36523559581/View-of-Delft/view_of_delft_detection_PUBLIC/view_of_delft_PUBLIC}
RAW=${RAW:-$DATA/crossmodal_fov70_refined_noaug_30ep_20261015/fixed_frames}
MILD=${MILD:-$DATA/stage2_mild_fog2_20frame_20261101}
OUT=${OUT:-$DATA/radar_pruner3vox_1p6m_mild_fog2_20261103}
SUPPORT=${SUPPORT:-$OUT/support_masks}

for required in "$PY" "$MILD/train/manifest.json" "$MILD/val/manifest.json" \
  "$RAW/train/manifest.json" "$RAW/val/manifest.json" \
  "$REPO/scripts/cache_radar_pruner_stage1.py" \
  "$REPO/scripts/audit_radar_pruner_geometry.py" \
  "$REPO/scripts/train_radar_pruner_stage1.py" \
  "$REPO/scripts/evaluate_radar_pruner_stage1.py"; do
  [[ -s "$required" ]] || { printf 'Missing required file: %s\n' "$required" >&2; exit 1; }
done
mkdir -p "$OUT"
cd "$REPO"
export PYTHONPATH="$REPO" OMP_NUM_THREADS=8
export LD_LIBRARY_PATH="$BASE/stage2-openblas-runtime/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

printf 'Step 1/4: cache 1.6 m radar-support masks.\n'
"$PY" -u -m scripts.cache_radar_pruner_stage1 \
  --vod-root "$VOD" --data-cache-root "$RAW" \
  --output-root "$SUPPORT" --split both --radius-m 1.6 \
  --num-workers 2 2>&1 | tee "$OUT/cache.log"

printf 'Step 2/4: audit eligible targets and full-recall ceiling.\n'
"$PY" -u -m scripts.audit_radar_pruner_geometry \
  --vod-root "$VOD" --data-cache-root "$RAW" \
  --faulty-root "$MILD" --support-cache-root "$SUPPORT" \
  --radius-m 1.6 --output "$OUT/geometry_val.json" --split val \
  2>&1 | tee "$OUT/geometry.log"

printf 'Step 3/4: train pruner for 90%% eligible recall.\n'
resume_args=()
if [[ -s "$OUT/last.ckpt" ]]; then
  resume_args=(--resume "$OUT/last.ckpt")
  printf 'Resuming from %s\n' "$OUT/last.ckpt"
fi
"$PY" -u -m scripts.train_radar_pruner_stage1 \
  --vod-root "$VOD" --data-cache-root "$RAW" \
  --faulty-root "$MILD" --support-cache-root "$SUPPORT" \
  --output-root "$OUT" --radius-m 1.6 \
  --target-eligible-recall 0.90 --epochs 25 --val-limit 64 \
  --num-workers 2 --device cuda "${resume_args[@]}" \
  2>&1 | tee "$OUT/train.log"

printf 'Step 4/4: 512-frame calibration and held-out evaluation.\n'
"$PY" -u -m scripts.evaluate_radar_pruner_stage1 \
  --vod-root "$VOD" --data-cache-root "$RAW" \
  --faulty-root "$MILD" --support-cache-root "$SUPPORT" \
  --checkpoint "$OUT/best.ckpt" \
  --output "$OUT/heldout_3vox_1p6m.json" \
  --limit 512 --device cuda \
  2>&1 | tee "$OUT/evaluate.log"

printf 'Completed: %s\n' "$OUT/heldout_3vox_1p6m.json"

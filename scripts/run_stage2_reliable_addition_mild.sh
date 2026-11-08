#!/usr/bin/env bash
# Train the addition head with standard BCE and a healthy-region penalty.
set -euo pipefail

REPO=${REPO:-/media/arrubuntu20/EA0A117F0A114A49/Thesis-1-stage2-geometry-16baaa0}
BASE=${BASE:-/mnt/3D10B36523559581/Gianluca}
DATA=${DATA:-/media/arrubuntu20/EA0A117F0A114A49/sensor_fusion_outputs}
PY=${PY:-$BASE/Sensor-Fusion/.venv_model_v2/bin/python}
VOD=${VOD:-/mnt/3D10B36523559581/View-of-Delft/view_of_delft_detection_PUBLIC/view_of_delft_PUBLIC}
RAW=${RAW:-$DATA/crossmodal_fov70_refined_noaug_30ep_20261015/fixed_frames}
SUPPORT=${SUPPORT:-$DATA/radar_support12vox_1p6m_mild_fog2_20261107}
MILD=${MILD:-$DATA/stage2_mild_fog2_20frame_20261101}
OLD=${OLD:-$DATA/stage2_clean_one_step_faulty_radar_12vox_1p6m_batch16_20261107}
OUT=${OUT:-$DATA/stage2_bce_healthy_addition_12vox_1p6m_20261108}
LPCR=${LPCR:-$DATA/radar_stage2_ray0_20261028/stage2_sparse_lpcr_fullregion_v9/best.ckpt}
if [[ ! -s "$LPCR" ]]; then
  LPCR=$DATA/stage2_fullregion_lpcr_for_geometry_20261106/best.ckpt
fi
SOURCE=$OLD/epoch125_snapshot/best_iou.ckpt
if [[ ! -s "$SOURCE" ]]; then
  SOURCE=$OLD/best_iou.ckpt
fi
for required in "$PY" "$LPCR" "$SOURCE" \
  "$RAW/train/manifest.json" "$RAW/val/manifest.json" \
  "$SUPPORT/train/manifest.json" "$SUPPORT/val/manifest.json" \
  "$MILD/train/manifest.json" "$MILD/val/manifest.json"; do
  [[ -s "$required" ]] || { printf 'Missing required file: %s\n' "$required" >&2; exit 1; }
done

mkdir -p "$OUT" "$DATA/stage2_clean_tmp"
if [[ ! -s "$OUT/base.ckpt" ]]; then
  cp "$SOURCE" "$OUT/base.ckpt"
fi
cd "$REPO"
export PYTHONPATH="$REPO" OMP_NUM_THREADS=8 CROSSMODAL_GPU_NEIGHBORS=0
export TMPDIR="$DATA/stage2_clean_tmp"
export LD_LIBRARY_PATH="$BASE/stage2-openblas-runtime/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

common=(--vod-root "$VOD" --data-cache-root "$RAW"
  --support-cache-root "$SUPPORT" --faulty-root "$MILD"
  --lpcr-checkpoint "$LPCR"
  --base-checkpoint "$OUT/base.ckpt" --batch-size 16
  --num-workers 2 --device cuda)

if [[ ! -s "$OUT/smoke/last.ckpt" ]]; then
  "$PY" -u -m scripts.train_stage2_reliable_addition \
    "${common[@]}" --output-root "$OUT/smoke" \
    --train-limit 32 --val-limit 8 --epochs 1 \
    2>&1 | tee "$OUT/smoke.log"
fi

resume=()
if [[ -s "$OUT/last.ckpt" ]]; then
  resume=(--resume "$OUT/last.ckpt")
fi
"$PY" -u -m scripts.train_stage2_reliable_addition \
  "${common[@]}" --output-root "$OUT" --epochs "${EPOCHS:-25}" \
  "${resume[@]}" 2>&1 | tee -a "$OUT/train.log"

if [[ "${EVAL:-1}" == 1 ]]; then
  for required in "$RAW/test/manifest.json" "$SUPPORT/test/manifest.json" \
    "$MILD/test/manifest.json" "$OUT/best_iou.ckpt"; do
    [[ -s "$required" ]] || { printf 'Missing evaluation file: %s\n' "$required" >&2; exit 1; }
  done
  "$PY" -u -m scripts.evaluate_stage2_reliable_addition \
    --vod-root "$VOD" --data-cache-root "$RAW" \
    --support-cache-root "$SUPPORT" --faulty-root "$MILD" \
    --lpcr-checkpoint "$LPCR" --base-checkpoint "$OUT/base.ckpt" \
    --addition-checkpoint "$OUT/best_iou.ckpt" \
    --output-json "$OUT/full_roi_best_iou_val_test.json" \
    --num-workers 2 --device cuda \
    2>&1 | tee "$OUT/full_roi_best_iou_val_test.log"
fi

printf 'History: %s\n' "$OUT/history.jsonl"
printf 'Full-ROI report: %s\n' "$OUT/full_roi_best_iou_val_test.json"

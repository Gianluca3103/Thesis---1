#!/usr/bin/env bash
# Restore clean latents from noisy faulty LiDAR; preserve Gaussian-start outputs.
set -euo pipefail

REPO=${REPO:-/media/arrubuntu20/EA0A117F0A114A49/Thesis-1-stage2-geometry-16baaa0}
BASE=${BASE:-/mnt/3D10B36523559581/Gianluca}
DATA=${DATA:-/media/arrubuntu20/EA0A117F0A114A49/sensor_fusion_outputs}
PY=${PY:-$BASE/Sensor-Fusion/.venv_model_v2/bin/python}
VOD=${VOD:-/mnt/3D10B36523559581/View-of-Delft/view_of_delft_detection_PUBLIC/view_of_delft_PUBLIC}
RAW=${RAW:-$DATA/crossmodal_fov70_refined_noaug_30ep_20261015/fixed_frames}
SUPPORT=${SUPPORT:-$DATA/radar_support12vox_1p6m_mild_fog2_20261107}
MILD=${MILD:-$DATA/stage2_mild_fog2_20frame_20261101}
FAULT_REGIONS=${FAULT_REGIONS:-}
OUT=${OUT:-$DATA/stage2_conditional_faulty_start_12vox_1p6m_20261108}
LPCR=${LPCR:-$DATA/radar_stage2_ray0_20261028/stage2_sparse_lpcr_fullregion_v9/best.ckpt}
BATCH_SIZE=${BATCH_SIZE:-16}
EPOCHS=${EPOCHS:-125}
TIMESTEP=${TIMESTEP:-499}
if [[ ! -s "$LPCR" ]]; then
  LPCR=$DATA/stage2_fullregion_lpcr_for_geometry_20261106/best.ckpt
fi
for required in "$PY" "$LPCR" \
  "$RAW/train/manifest.json" "$RAW/val/manifest.json" "$RAW/test/manifest.json" \
  "$SUPPORT/train/manifest.json" "$SUPPORT/val/manifest.json" \
  "$SUPPORT/test/manifest.json" "$MILD/train/manifest.json" \
  "$MILD/val/manifest.json" "$MILD/test/manifest.json"; do
  [[ -s "$required" ]] || { printf 'Missing required file: %s\n' "$required" >&2; exit 1; }
done
region_arg=()
if [[ -n "$FAULT_REGIONS" ]]; then
  for split in train val test; do
    [[ -s "$FAULT_REGIONS/$split/regions.json" ]] || {
      printf 'Missing injector footprints: %s\n' "$FAULT_REGIONS/$split/regions.json" >&2
      exit 1
    }
  done
  region_arg=(--fault-region-root "$FAULT_REGIONS")
fi

mkdir -p "$OUT" "$DATA/stage2_clean_tmp"
cd "$REPO"
export PYTHONPATH="$REPO" OMP_NUM_THREADS=8 CROSSMODAL_GPU_NEIGHBORS=0
export TMPDIR="$DATA/stage2_clean_tmp"
export LD_LIBRARY_PATH="$BASE/stage2-openblas-runtime/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

common=(--vod-root "$VOD" --data-cache-root "$RAW"
  --support-cache-root "$SUPPORT" --faulty-root "$MILD"
  "${region_arg[@]}"
  --lpcr-checkpoint "$LPCR" --batch-size "$BATCH_SIZE"
  --num-workers 2 --device cuda)

printf 'Smoke test: 32 train / 8 validation frames; faulty-latent one-step.\n'
if [[ ! -s "$OUT/smoke/last.ckpt" ]]; then
  "$PY" -u -m scripts.train_stage2_conditional_faulty_start \
    "${common[@]}" --output-root "$OUT/smoke" \
    --train-limit 32 --val-limit 8 --eval-timestep "$TIMESTEP" --epochs 1 \
    2>&1 | tee "$OUT/smoke.log"
fi

resume=()
if [[ -s "$OUT/last.ckpt" ]]; then
  resume=(--resume "$OUT/last.ckpt")
fi
printf 'Training %s epochs; full one-step faulty-start validation every epoch.\n' \
  "$EPOCHS"
"$PY" -u -m scripts.train_stage2_conditional_faulty_start \
  "${common[@]}" --output-root "$OUT" --epochs "$EPOCHS" \
  --eval-timestep "$TIMESTEP" "${resume[@]}" \
  2>&1 | tee -a "$OUT/train.log"

printf 'Scoring the best validation-IoU checkpoint on official validation and test.\n'
"$PY" -u -m scripts.evaluate_stage2_conditional_reconstruction \
  --vod-root "$VOD" --data-cache-root "$RAW" \
  --support-cache-root "$SUPPORT" --faulty-root "$MILD" \
  "${region_arg[@]}" \
  --lpcr-checkpoint "$LPCR" --checkpoint "$OUT/best_iou.ckpt" \
  --steps 1 --num-workers 2 --device cuda \
  --output-json "$OUT/best_faulty_start_1step_val_test.json" \
  2>&1 | tee "$OUT/best_faulty_start_1step_val_test.log"

printf 'History: %s\n' "$OUT/history.jsonl"
printf 'Report: %s\n' "$OUT/best_faulty_start_1step_val_test.json"

GAUSSIAN_REPORT=${GAUSSIAN_REPORT:-$DATA/stage2_conditional_scratch_12vox_1p6m_20261108/best_full_noise_64steps_val_test.json}
if [[ -s "$GAUSSIAN_REPORT" ]]; then
  "$PY" -u -m scripts.compare_stage2_input_sources \
    --gaussian-report "$GAUSSIAN_REPORT" \
    --faulty-start-report "$OUT/best_faulty_start_1step_val_test.json" \
    --output-json "$OUT/comparison_with_gaussian_start.json" \
    2>&1 | tee "$OUT/comparison_with_gaussian_start.log"
else
  printf 'Gaussian-start report not found yet: %s\n' "$GAUSSIAN_REPORT"
fi

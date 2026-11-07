#!/usr/bin/env bash
# Evaluate the one-step faulty/radar-conditioned diagnostic on val and test.
set -euo pipefail

REPO=${REPO:-/media/arrubuntu20/EA0A117F0A114A49/Thesis-1-stage2-geometry-16baaa0}
BASE=${BASE:-/mnt/3D10B36523559581/Gianluca}
DATA=${DATA:-/media/arrubuntu20/EA0A117F0A114A49/sensor_fusion_outputs}
PY=${PY:-$BASE/Sensor-Fusion/.venv_model_v2/bin/python}
VOD=${VOD:-/mnt/3D10B36523559581/View-of-Delft/view_of_delft_detection_PUBLIC/view_of_delft_PUBLIC}
RAW=${RAW:-$DATA/crossmodal_fov70_refined_noaug_30ep_20261015/fixed_frames}
SUPPORT=${SUPPORT:-$DATA/radar_support12vox_1p6m_mild_fog2_20261107}
MILD=${MILD:-$DATA/stage2_mild_fog2_20frame_20261101}
FOG_SIM=${FOG_SIM:-$BASE/Sensor-Fusion/Weather_Injector/LiDAR_fog_sim}
OUT=${OUT:-$DATA/stage2_clean_one_step_faulty_radar_12vox_1p6m_batch16_20261107}
LPCR=${LPCR:-$DATA/radar_stage2_ray0_20261028/stage2_sparse_lpcr_fullregion_v9/best.ckpt}
if [[ ! -s "$LPCR" ]]; then
  LPCR=$DATA/stage2_fullregion_lpcr_for_geometry_20261106/best.ckpt
fi
for required in "$PY" "$LPCR" "$OUT/best.ckpt" "$OUT/best_iou.ckpt" \
  "$OUT/last.ckpt" "$RAW/train/manifest.json" "$RAW/val/manifest.json" \
  "$RAW/test/manifest.json" "$SUPPORT/train/manifest.json" \
  "$SUPPORT/val/manifest.json" "$SUPPORT/test/manifest.json" \
  "$MILD/train/manifest.json" "$MILD/val/manifest.json" \
  "$FOG_SIM/fog_simulation.py"; do
  [[ -s "$required" ]] || { printf 'Missing required file: %s\n' "$required" >&2; exit 1; }
done
completed=$("$PY" -c 'import sys,torch; print(torch.load(sys.argv[1], map_location="cpu", weights_only=False)["epoch"])' "$OUT/last.ckpt")
if (( completed < 125 )); then
  printf 'Conditioned trial is at epoch %s; wait for epoch 125.\n' "$completed" >&2
  exit 1
fi
cd "$REPO"
mkdir -p "$DATA/stage2_clean_tmp"
export PYTHONPATH="$REPO" OMP_NUM_THREADS=8 CROSSMODAL_GPU_NEIGHBORS=0
export TMPDIR="$DATA/stage2_clean_tmp"
export LD_LIBRARY_PATH="$BASE/stage2-openblas-runtime/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

printf 'Caching deterministic mild faults for the official test split.\n'
"$PY" -u -m scripts.cache_stage2_mild_fog2 \
  --vod-root "$VOD" --data-cache-root "$RAW" \
  --fog-simulator-root "$FOG_SIM" --output-root "$MILD" \
  --split test --num-workers 2 --max-cache-gib 5 \
  2>&1 | tee "$OUT/cache_test_mild.log"

common=(--vod-root "$VOD" --data-cache-root "$RAW"
  --support-cache-root "$SUPPORT" --faulty-root "$MILD"
  --lpcr-checkpoint "$LPCR" --num-workers 2 --device cuda)
for name in best best_iou last; do
  checkpoint="$OUT/$name.ckpt"
  sweep="$OUT/threshold_sweep_conditioned_$name.json"
  "$PY" -u -m scripts.evaluate_stage2_clean_denoiser_thresholds \
    "${common[@]}" --checkpoint "$checkpoint" \
    --sampling-strategy one_step --output-json "$sweep" \
    2>&1 | tee "$OUT/threshold_sweep_conditioned_$name.log"
  for split in val test; do
    "$PY" -u -m scripts.evaluate_stage2_clean_selected_metrics \
      "${common[@]}" --checkpoint "$checkpoint" --sweep-json "$sweep" \
      --sampling-strategy one_step --split "$split" \
      --output-json "$OUT/selected_conditioned_${name}_${split}.json" \
      2>&1 | tee "$OUT/selected_conditioned_${name}_${split}.log"
  done
done

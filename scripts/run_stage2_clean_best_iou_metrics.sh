#!/usr/bin/env bash
# Select best exact IoU threshold, then score tolerant IoU/F1 and Chamfer.
set -euo pipefail

REPO=${REPO:-/media/arrubuntu20/EA0A117F0A114A49/Thesis-1-stage2-geometry-16baaa0}
BASE=${BASE:-/mnt/3D10B36523559581/Gianluca}
DATA=${DATA:-/media/arrubuntu20/EA0A117F0A114A49/sensor_fusion_outputs}
PY=${PY:-$BASE/Sensor-Fusion/.venv_model_v2/bin/python}
VOD=${VOD:-/mnt/3D10B36523559581/View-of-Delft/view_of_delft_detection_PUBLIC/view_of_delft_PUBLIC}
RAW=${RAW:-$DATA/crossmodal_fov70_refined_noaug_30ep_20261015/fixed_frames}
SUPPORT=${SUPPORT:-$DATA/radar_support12vox_1p6m_mild_fog2_20261107}
OUT=${OUT:-$DATA/stage2_clean_denoiser_12vox_1p6m_batch16_20261107}
LPCR=${LPCR:-$DATA/radar_stage2_ray0_20261028/stage2_sparse_lpcr_fullregion_v9/best.ckpt}
if [[ ! -s "$LPCR" ]]; then
  LPCR=$DATA/stage2_fullregion_lpcr_for_geometry_20261106/best.ckpt
fi
for required in "$PY" "$LPCR" "$OUT/best.ckpt" "$OUT/last.ckpt" \
  "$RAW/train/manifest.json" "$RAW/val/manifest.json" \
  "$SUPPORT/train/manifest.json" "$SUPPORT/val/manifest.json"; do
  [[ -s "$required" ]] || { printf 'Missing required file: %s\n' "$required" >&2; exit 1; }
done
mkdir -p "$DATA/stage2_clean_tmp"
cd "$REPO"
export PYTHONPATH="$REPO" OMP_NUM_THREADS=8 CROSSMODAL_GPU_NEIGHBORS=0
export TMPDIR="$DATA/stage2_clean_tmp"
export LD_LIBRARY_PATH="$BASE/stage2-openblas-runtime/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

printf 'Preparing official test radar with the training radar policy.\n'
"$PY" -u -m scripts.generate_vod_accumulated_radar \
  --vod-root "$VOD" --stack-sizes 20 --split test \
  --alignment-source official-5 --output-suffix verified \
  --isolate-split-history --no-basic-validity-filter --num-workers 2 \
  2>&1 | tee "$OUT/prepare_test_verified_radar.log"
"$PY" -u -m scripts.generate_vod_doppler_radar \
  --vod-root "$VOD" --split test \
  --output-variant radar_20frames_verified_doppler_radial \
  --frame-period-s 0.10022 --doppler-sign 1 --num-workers 2 \
  2>&1 | tee "$OUT/prepare_test_doppler_radar.log"

printf 'Caching clean test frames and the 12-radar-voxel/1.6 m gate.\n'
"$PY" -u -m scripts.cache_crossmodal_fixed_frames \
  --vod-root "$VOD" --cache-root "$RAW" --split test \
  --num-workers 2 --minimum-free-gib 2 \
  2>&1 | tee "$OUT/cache_test_fixed_frames.log"
"$PY" -u -m scripts.cache_radar_pruner_stage1 \
  --vod-root "$VOD" --data-cache-root "$RAW" \
  --output-root "$SUPPORT" --split test --radius-m 1.6 \
  --min-distinct-radar-voxels 12 --num-workers 2 \
  2>&1 | tee "$OUT/cache_test_support.log"

common=(--vod-root "$VOD" --data-cache-root "$RAW"
  --support-cache-root "$SUPPORT" --lpcr-checkpoint "$LPCR"
  --num-workers 2 --device cuda)
for name in best last; do
  checkpoint="$OUT/$name.ckpt"
  sweep="$OUT/threshold_sweep_clean_$name.json"
  printf 'Building %s threshold sweep on every validation frame.\n' "$name"
  "$PY" -u -m scripts.evaluate_stage2_clean_denoiser_thresholds \
    "${common[@]}" --checkpoint "$checkpoint" --output-json "$sweep" \
    2>&1 | tee "$OUT/threshold_sweep_clean_$name.log"
  for split in val test; do
    if [[ "$split" == val ]]; then
      report="$OUT/selected_occupancy_$name.json"
      log="$OUT/selected_occupancy_$name.log"
    else
      report="$OUT/selected_occupancy_${name}_test.json"
      log="$OUT/selected_occupancy_${name}_test.log"
    fi
    printf 'Scoring %s on %s at validation-selected best exact IoU threshold.\n' \
      "$name" "$split"
    "$PY" -u -m scripts.evaluate_stage2_clean_selected_metrics \
      "${common[@]}" --checkpoint "$checkpoint" --sweep-json "$sweep" \
      --split "$split" --output-json "$report" \
      2>&1 | tee "$log"
  done
done

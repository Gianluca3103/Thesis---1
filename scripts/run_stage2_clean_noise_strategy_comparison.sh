#!/usr/bin/env bash
# Cross the old/new gate-only checkpoints with one-step and three-step sampling.
set -euo pipefail

REPO=${REPO:-/media/arrubuntu20/EA0A117F0A114A49/Thesis-1-stage2-geometry-16baaa0}
BASE=${BASE:-/mnt/3D10B36523559581/Gianluca}
DATA=${DATA:-/media/arrubuntu20/EA0A117F0A114A49/sensor_fusion_outputs}
PY=${PY:-$BASE/Sensor-Fusion/.venv_model_v2/bin/python}
VOD=${VOD:-/mnt/3D10B36523559581/View-of-Delft/view_of_delft_detection_PUBLIC/view_of_delft_PUBLIC}
RAW=${RAW:-$DATA/crossmodal_fov70_refined_noaug_30ep_20261015/fixed_frames}
SUPPORT=${SUPPORT:-$DATA/radar_support12vox_1p6m_mild_fog2_20261107}
OLD=${OLD:-$DATA/stage2_clean_denoiser_12vox_1p6m_batch16_20261107}
NEW=${NEW:-$DATA/stage2_clean_fullnoise_12vox_1p6m_5train_3val_20261107}
OUT=${OUT:-$DATA/stage2_clean_noise_strategy_comparison_20261107}
LPCR=${LPCR:-$DATA/radar_stage2_ray0_20261028/stage2_sparse_lpcr_fullregion_v9/best.ckpt}
if [[ ! -s "$LPCR" ]]; then
  LPCR=$DATA/stage2_fullregion_lpcr_for_geometry_20261106/best.ckpt
fi
for required in "$PY" "$LPCR" "$OLD/best.ckpt" "$NEW/best.ckpt" \
  "$NEW/best_iou.ckpt" \
  "$NEW/last.ckpt" "$RAW/train/manifest.json" "$RAW/val/manifest.json" \
  "$RAW/test/manifest.json" "$SUPPORT/train/manifest.json" \
  "$SUPPORT/val/manifest.json" "$SUPPORT/test/manifest.json"; do
  [[ -s "$required" ]] || { printf 'Missing required file: %s\n' "$required" >&2; exit 1; }
done
completed=$("$PY" -c 'import sys,torch; print(torch.load(sys.argv[1], map_location="cpu", weights_only=False)["epoch"])' "$NEW/last.ckpt")
if (( completed < 125 )); then
  printf 'Full-noise trial is at epoch %s; wait for epoch 125.\n' "$completed" >&2
  exit 1
fi
mkdir -p "$OUT" "$DATA/stage2_clean_tmp"
cd "$REPO"
export PYTHONPATH="$REPO" OMP_NUM_THREADS=8 CROSSMODAL_GPU_NEIGHBORS=0
export TMPDIR="$DATA/stage2_clean_tmp"
export LD_LIBRARY_PATH="$BASE/stage2-openblas-runtime/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

common=(--vod-root "$VOD" --data-cache-root "$RAW"
  --support-cache-root "$SUPPORT" --lpcr-checkpoint "$LPCR"
  --num-workers 2 --device cuda)
for model in old new new_iou; do
  if [[ "$model" == old ]]; then
    checkpoint="$OLD/best.ckpt"
  elif [[ "$model" == new_iou ]]; then
    checkpoint="$NEW/best_iou.ckpt"
  else
    checkpoint="$NEW/best.ckpt"
  fi
  for strategy in one_step reverse3; do
    name="${model}_${strategy}"
    sweep="$OUT/${name}_sweep.json"
    printf 'Sweeping %s on all validation frames.\n' "$name"
    "$PY" -u -m scripts.evaluate_stage2_clean_denoiser_thresholds \
      "${common[@]}" --checkpoint "$checkpoint" \
      --sampling-strategy "$strategy" --output-json "$sweep" \
      2>&1 | tee "$OUT/${name}_sweep.log"
    for split in val test; do
      printf 'Scoring %s on %s with its calibration-selected threshold.\n' \
        "$name" "$split"
      "$PY" -u -m scripts.evaluate_stage2_clean_selected_metrics \
        "${common[@]}" --checkpoint "$checkpoint" \
        --sampling-strategy "$strategy" --sweep-json "$sweep" \
        --split "$split" --output-json "$OUT/${name}_${split}.json" \
        2>&1 | tee "$OUT/${name}_${split}.log"
    done
  done
done
"$PY" -u -m scripts.summarize_stage2_clean_noise_strategy_comparison \
  --report-root "$OUT"

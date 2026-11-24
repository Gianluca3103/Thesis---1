#!/usr/bin/env bash
# Fresh PointNet++ radar AE and LiDAR LPCR, followed by partial-loss Stage II.
set -euo pipefail

PHASE=${1:-all}
BASE=${BASE:-/mnt/3D10B36523559581/Gianluca}
DATA=${DATA:-$BASE/sensor_fusion_outputs/sensor_fusion_outputs}
REPO=${REPO:-$BASE/sensor_fusion_outputs/Thesis-1-radar-autoencoder-prefilter-20261109}
PY=${PY:-$BASE/Sensor-Fusion/.venv_model_v2/bin/python}
VOD=${VOD:-/mnt/3D10B36523559581/View-of-Delft/view_of_delft_detection_PUBLIC/view_of_delft_PUBLIC}
UPSTREAM=${UPSTREAM:-$DATA/radar_dpcc_official_12vox_1p6m_20261111/upstream/D-PCC}
OUT=${OUT:-$DATA/e2e_pointnetpp1024_lpcr_stage2_partial_only_20261124}
RAW=${RAW:-$OUT/fixed_frames}
MILD=${MILD:-$OUT/partial_loss_faults}
SUPPORT=${SUPPORT:-$OUT/radar_support12vox_1p6m}
REGIONS=${REGIONS:-$OUT/fault_regions}
RADAR_CACHE=$OUT/radar_filtered_cache
RADAR_OUT=$OUT/radar_autoencoder
RADAR_FEATURES=$OUT/radar_encoder_features
LPCR_OUT=$OUT/lidar_lpcr
STAGE2_OUT=$OUT/stage2
RADAR_EPOCHS=${RADAR_EPOCHS:-300}
LPCR_EPOCHS=${LPCR_EPOCHS:-30}
STAGE2_EPOCHS=${STAGE2_EPOCHS:-75}
WORKERS=${WORKERS:-2}

[[ -x "$PY" && -d "$VOD" && -s "$REPO/scripts/train_pointnetpp_stage2_radar.py" ]] || {
  echo 'Missing Python, VoD dataset, or pinned repository files' >&2; exit 1;
}
mkdir -p "$OUT" "$BASE/rtmp"
export TMPDIR="$BASE/rtmp" OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=1
export CROSSMODAL_GPU_NEIGHBORS=0
export LD_LIBRARY_PATH="$BASE/stage2-openblas-runtime/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PYTHONPATH="$REPO:$UPSTREAM/models:$UPSTREAM/models/pointops${PYTHONPATH:+:$PYTHONPATH}"
cd "$REPO"

require_pointops() {
  [[ -s "$UPSTREAM/models/pointops/functions/pointops.py" ]] || {
    echo "Missing compiled D-PCC pointops tree: $UPSTREAM" >&2; exit 1;
  }
  "$PY" -c 'import torch; import pointops_cuda; from pointops.functions import pointops; print("PointNet++ FPS ready")'
}

prepare() {
  # The shared VoD radar variant is generated if missing and verified if present.
  for split in train val test; do
    "$PY" -u -m scripts.generate_vod_accumulated_radar \
      --vod-root "$VOD" --stack-sizes 20 --split "$split" \
      --alignment-source official-5 --output-suffix verified \
      --isolate-split-history --no-basic-validity-filter \
      --num-workers "$WORKERS" 2>&1 | tee -a "$OUT/radar_stacks.log"
    "$PY" -u -m scripts.generate_vod_doppler_radar \
      --vod-root "$VOD" --split "$split" \
      --output-variant radar_20frames_verified_doppler_radial \
      --frame-period-s 0.10022 --doppler-sign 1 \
      --num-workers "$WORKERS" 2>&1 | tee -a "$OUT/radar_stacks.log"
  done
  "$PY" -u -m scripts.check_crossmodal_inputs \
    --vod-root "$VOD" \
    --radar-variant radar_20frames_verified_doppler_radial \
    2>&1 | tee -a "$OUT/radar_stacks.log"
  for split in train val test; do
    "$PY" -u -m scripts.cache_crossmodal_fixed_frames \
      --vod-root "$VOD" --cache-root "$RAW" --split "$split" \
      --radar-variant radar_20frames_verified_doppler_radial \
      --num-workers "$WORKERS" 2>&1 | tee -a "$OUT/raw_cache.log"
  done
  for split in both test; do
    "$PY" -u -m scripts.cache_radar_pruner_stage1 \
      --vod-root "$VOD" --data-cache-root "$RAW" \
      --output-root "$SUPPORT" --split "$split" --radius-m 1.6 \
      --min-distinct-radar-voxels 12 --num-workers "$WORKERS" \
      2>&1 | tee -a "$OUT/support_cache.log"
  done
  "$PY" -u -m scripts.cache_stage2_partial_loss \
    --vod-root "$VOD" --data-cache-root "$RAW" \
    --output-root "$MILD" --split all --num-workers "$WORKERS" \
    2>&1 | tee -a "$OUT/partial_loss_cache.log"
  for split in train val test; do
    "$PY" -u -m scripts.pointnetpp_stage2_radar_data \
      --vod-root "$VOD" --cache-root "$RADAR_CACHE" --split "$split" \
      2>&1 | tee -a "$OUT/radar_filter_cache.log"
  done
}

radar() {
  require_pointops
  [[ -s "$RADAR_CACHE/train/manifest.json" && -s "$RADAR_CACHE/val/manifest.json" ]] || {
    echo 'Build filtered radar cache with prepare phase first' >&2; exit 1;
  }
  mkdir -p "$RADAR_OUT"
  "$PY" -u -m scripts.train_pointnetpp_stage2_radar \
    --variant pointnetpp --cache-root "$RADAR_CACHE" \
    --output-root "$RADAR_OUT" --epochs "$RADAR_EPOCHS" \
    --batch-size 4 --workers "$WORKERS" --input-points 2048 \
    --anchors 512 --width 1024 --children 4 --dropout-p 0 \
    2>&1 | tee -a "$OUT/radar_train.log"
  [[ -s "$RADAR_OUT/best_iou.ckpt" ]] || { echo 'Radar AE has no best checkpoint' >&2; exit 1; }
  for split in val test; do
    "$PY" -u -m scripts.train_pointnetpp_stage2_radar --phase evaluate \
      --variant pointnetpp --cache-root "$RADAR_CACHE" \
      --output-root "$RADAR_OUT" --checkpoint "$RADAR_OUT/best_iou.ckpt" \
      --split "$split" --workers "$WORKERS" \
      2>&1 | tee -a "$OUT/radar_eval_$split.log"
  done
}

lidar() {
  [[ -s "$RAW/train/manifest.json" && -s "$RAW/val/manifest.json" ]] || {
    echo 'Build clean LiDAR cache with prepare phase first' >&2; exit 1;
  }
  mkdir -p "$LPCR_OUT"
  resume=()
  [[ ! -s "$LPCR_OUT/last.ckpt" ]] || resume=(--resume "$LPCR_OUT/last.ckpt")
  "$PY" -u -m scripts.train_stage2_sparse --phase lpcr \
    --vod-root "$VOD" --data-cache-root "$RAW" \
    --output-root "$LPCR_OUT" \
    --radar-variant radar_20frames_verified_doppler_radial \
    --epochs "$LPCR_EPOCHS" --lr 0.001 --batch-size 8 \
    --num-workers "$WORKERS" --paper-metrics-every 5 \
    --paper-metrics-frames 4 --device cuda "${resume[@]}" \
    2>&1 | tee -a "$OUT/lidar_lpcr_train.log"
  [[ -s "$LPCR_OUT/best.ckpt" ]] || { echo 'LiDAR LPCR has no best checkpoint' >&2; exit 1; }
}

features() {
  require_pointops
  [[ -s "$RADAR_OUT/best_iou.ckpt" ]] || { echo 'Train radar AE first' >&2; exit 1; }
  for split in train val test; do
    "$PY" -u -m scripts.cache_pointnetpp_stage2_features \
      --radar-cache-root "$RADAR_CACHE" \
      --radar-checkpoint "$RADAR_OUT/best_iou.ckpt" \
      --output-root "$RADAR_FEATURES" --split "$split" --device cuda \
      2>&1 | tee -a "$OUT/radar_features.log"
  done
}

stage2() {
  [[ -s "$LPCR_OUT/best.ckpt" && -s "$RADAR_OUT/best_iou.ckpt" &&
     -s "$RADAR_FEATURES/test/manifest.json" ]] || {
    echo 'Train both encoders and cache radar features first' >&2; exit 1;
  }
  REPO="$REPO" BASE="$BASE" DATA="$DATA" PY="$PY" VOD="$VOD" \
    RAW="$RAW" SUPPORT="$SUPPORT" MILD="$MILD" \
    FAULT_REGIONS="$REGIONS" LPCR="$LPCR_OUT/best.ckpt" \
    RADAR_ENCODER="$RADAR_OUT/best_iou.ckpt" RADAR_FEATURES="$RADAR_FEATURES" \
    OUT="$STAGE2_OUT" GENERATED_BCE=1 BCE_LOSS_WEIGHT=1.0 \
    DECODER_LOSS=0 FIXED_VAL_ONLY=0 INIT='' \
    BATCH_SIZE=16 WORKERS="$WORKERS" EPOCHS="$STAGE2_EPOCHS" \
    ROLLOUT_STEPS=64 GRADIENT_STEPS=4 VAL_STEPS=64 RECON_VAL_EVERY=5 \
    bash "$REPO/scripts/run_stage2_conditional_from_scratch.sh"
}

case "$PHASE" in
  prepare) prepare ;;
  radar) radar ;;
  lidar) lidar ;;
  features) features ;;
  stage2) stage2 ;;
  all) require_pointops; prepare; radar; lidar; features; stage2 ;;
  *) echo "Usage: bash $0 [prepare|radar|lidar|features|stage2|all]" >&2; exit 2 ;;
esac

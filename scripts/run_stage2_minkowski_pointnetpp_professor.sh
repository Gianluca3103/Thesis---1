#!/usr/bin/env bash
# Width-768 PointNet++ radar conditioning and frozen Minkowski LiDAR decoder.
set -euo pipefail

PHASE=${1:-all}
BASE=${BASE:-/mnt/3D10B36523559581/Gianluca}
DATA=${DATA:-$BASE/sensor_fusion_outputs/sensor_fusion_outputs}
REPO=${REPO:-$BASE/sensor_fusion_outputs/Thesis-1-stage2-minkowski-pointnet768}
PY=${PY:-$BASE/Sensor-Fusion/.venv_model_v2/bin/python}
VOD=${VOD:-/mnt/3D10B36523559581/View-of-Delft/view_of_delft_detection_PUBLIC/view_of_delft_PUBLIC}
UPSTREAM=${UPSTREAM:-$DATA/radar_dpcc_official_12vox_1p6m_20261111/upstream/D-PCC}
OUT=${OUT:-$DATA/stage2_minkowski_pointnet768_partial_only_20261125}
CACHE_OUT=${CACHE_OUT:-$DATA/e2e_pointnetpp1024_lpcr_stage2_partial_only_20261124}
RAW=${RAW:-$CACHE_OUT/fixed_frames}
SUPPORT=${SUPPORT:-$CACHE_OUT/radar_support12vox_1p6m}
MILD=${MILD:-$CACHE_OUT/partial_loss_faults}
REGIONS=${REGIONS:-$CACHE_OUT/fault_regions}
RADAR_CACHE=${RADAR_CACHE:-$CACHE_OUT/radar_filtered_cache}
LIDAR_CKPT=${LIDAR_CKPT:-$BASE/sensor_fusion_outputs/minkowski_lidar_ae_stage2grid_20261125/model/best.ckpt}
if [[ ! -s "$LIDAR_CKPT" && -s "$DATA/minkowski_lidar_ae_stage2grid_20261125/model/best.ckpt" ]]; then
  LIDAR_CKPT="$DATA/minkowski_lidar_ae_stage2grid_20261125/model/best.ckpt"
fi
RADAR_OUT=$OUT/pointnetpp768
RADAR_FEATURES=$OUT/radar_features
STAGE2_OUT=$OUT/stage2
RADAR_EPOCHS=${RADAR_EPOCHS:-300}
STAGE2_EPOCHS=${STAGE2_EPOCHS:-75}
RADAR_BATCH=${RADAR_BATCH:-4}
STAGE2_BATCH=${STAGE2_BATCH:-4}
WORKERS=${WORKERS:-4}
ROLLOUT_STEPS=${ROLLOUT_STEPS:-64}
GRADIENT_STEPS=${GRADIENT_STEPS:-4}
VAL_STEPS=${VAL_STEPS:-64}

[[ -x "$PY" && -d "$VOD" && -s "$REPO/scripts/train_stage2_minkowski_completion.py" ]] || {
  echo 'Missing professor Python, VoD data, or Stage II repository' >&2; exit 1;
}
mkdir -p "$OUT" "$BASE/rtmp"
TORCH_LIB=$("$PY" -c 'import torch; from pathlib import Path; print(Path(torch.__file__).parent / "lib")')
export LD_LIBRARY_PATH="$TORCH_LIB:$BASE/stage2-openblas-runtime/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PYTHONPATH="$REPO:$UPSTREAM/models:$UPSTREAM/models/pointops${PYTHONPATH:+:$PYTHONPATH}"
export TMPDIR="$BASE/rtmp" OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=1 CROSSMODAL_GPU_NEIGHBORS=0
cd "$REPO"

check_pointnet() {
  [[ -s "$UPSTREAM/models/pointops/functions/pointops.py" ]] || {
    echo "Missing compiled PointNet++ operator source: $UPSTREAM" >&2; exit 1;
  }
  "$PY" -c 'import torch, pointops_cuda; from pointops.functions import pointops; print("PointNet++ CUDA ready")'
}

prepare() {
  # Reuses the established full train/val/test caches when already present.
  REPO="$REPO" BASE="$BASE" DATA="$DATA" PY="$PY" VOD="$VOD" \
    UPSTREAM="$UPSTREAM" OUT="$CACHE_OUT" RAW="$RAW" SUPPORT="$SUPPORT" \
    MILD="$MILD" WORKERS="$WORKERS" \
    bash "$REPO/scripts/run_pointnetpp_lpcr_stage2_professor.sh" prepare
  "$PY" -u -m scripts.cache_stage2_fault_regions \
    --vod-root "$VOD" --data-cache-root "$RAW" \
    --faulty-root "$MILD" --output-root "$REGIONS" --split all
}

radar() {
  check_pointnet
  [[ -s "$RADAR_CACHE/train/manifest.json" && -s "$RADAR_CACHE/val/manifest.json" ]] || {
    echo 'Run prepare first to build the filtered 20-sweep radar cache' >&2; exit 1;
  }
  mkdir -p "$RADAR_OUT"
  "$PY" -u -m scripts.train_pointnetpp_stage2_radar \
    --variant pointnetpp --cache-root "$RADAR_CACHE" \
    --output-root "$RADAR_OUT" --epochs "$RADAR_EPOCHS" \
    --batch-size "$RADAR_BATCH" --workers "$WORKERS" \
    --input-points 2048 --anchors 512 --width 768 --children 4 \
    --dropout-p 0 2>&1 | tee -a "$OUT/radar_train.log"
  [[ -s "$RADAR_OUT/best_iou.ckpt" ]] || {
    echo 'Radar encoder did not produce best_iou.ckpt' >&2; exit 1;
  }
  for split in val test; do
    "$PY" -u -m scripts.train_pointnetpp_stage2_radar \
      --phase evaluate --variant pointnetpp \
      --cache-root "$RADAR_CACHE" --output-root "$RADAR_OUT" \
      --checkpoint "$RADAR_OUT/best_iou.ckpt" --split "$split" \
      --workers "$WORKERS" 2>&1 | tee -a "$OUT/radar_eval_$split.log"
  done
}

features() {
  check_pointnet
  [[ -s "$RADAR_OUT/best_iou.ckpt" ]] || { echo 'Run radar first' >&2; exit 1; }
  for split in train val test; do
    "$PY" -u -m scripts.cache_pointnetpp_stage2_features \
      --radar-cache-root "$RADAR_CACHE" \
      --radar-checkpoint "$RADAR_OUT/best_iou.ckpt" \
      --output-root "$RADAR_FEATURES" --split "$split" --device cuda \
      2>&1 | tee -a "$OUT/radar_features.log"
  done
}

check_stage2() {
  for required in "$LIDAR_CKPT" "$RADAR_OUT/best_iou.ckpt" \
      "$RADAR_FEATURES/train/manifest.json" \
      "$RADAR_FEATURES/val/manifest.json" \
      "$RADAR_FEATURES/test/manifest.json"; do
    [[ -s "$required" ]] || { echo "Missing Stage II dependency: $required" >&2; exit 1; }
  done
  for split in train val test; do
    for required in "$RAW/$split/manifest.json" \
        "$SUPPORT/$split/manifest.json" "$MILD/$split/manifest.json" \
        "$REGIONS/$split/manifest.json"; do
      [[ -s "$required" ]] || { echo "Missing Stage II cache: $required" >&2; exit 1; }
    done
  done
  "$PY" -c 'import torch, MinkowskiEngine as ME; print("MinkowskiEngine CUDA ready:", torch.cuda.get_device_name())'
}

common=(--vod-root "$VOD" --data-cache-root "$RAW" \
  --support-cache-root "$SUPPORT" --faulty-root "$MILD" \
  --fault-region-root "$REGIONS" --radar-feature-root "$RADAR_FEATURES" \
  --radar-checkpoint "$RADAR_OUT/best_iou.ckpt" \
  --lidar-checkpoint "$LIDAR_CKPT" --workers "$WORKERS")

smoke() {
  check_stage2
  if [[ -s "$OUT/smoke/last.ckpt" ]]; then
    echo 'Smoke test already passed; skipping it'
    return
  fi
  "$PY" -u -m scripts.train_stage2_minkowski_completion \
    "${common[@]}" --output-root "$OUT/smoke" \
    --epochs 1 --batch-size "$STAGE2_BATCH" --train-limit 16 --val-limit 8 \
    --rollout-steps 4 --gradient-steps 2 --validation-steps 4 \
    --calibration-frames 2 --skip-final-evaluation \
    2>&1 | tee "$OUT/smoke.log"
}

stage2() {
  check_stage2
  mkdir -p "$STAGE2_OUT"
  resume=()
  [[ ! -s "$STAGE2_OUT/last.ckpt" ]] || resume=(--resume "$STAGE2_OUT/last.ckpt")
  "$PY" -u -m scripts.train_stage2_minkowski_completion \
    "${common[@]}" --output-root "$STAGE2_OUT" \
    --epochs "$STAGE2_EPOCHS" --batch-size "$STAGE2_BATCH" \
    --rollout-steps "$ROLLOUT_STEPS" --gradient-steps "$GRADIENT_STEPS" \
    --validation-steps "$VAL_STEPS" --reconstruction-val-every 5 \
    --bce-weight 1.0 "${resume[@]}" 2>&1 | tee -a "$OUT/stage2_train.log"
}

evaluate() {
  check_stage2
  [[ -s "$STAGE2_OUT/best_iou.ckpt" ]] || { echo 'Train Stage II first' >&2; exit 1; }
  "$PY" -u -m scripts.train_stage2_minkowski_completion \
    "${common[@]}" --phase evaluate --output-root "$STAGE2_OUT" \
    --checkpoint "$STAGE2_OUT/best_iou.ckpt" \
    --batch-size "$STAGE2_BATCH" --rollout-steps "$ROLLOUT_STEPS" \
    --gradient-steps "$GRADIENT_STEPS" --validation-steps "$VAL_STEPS" \
    --bce-weight 1.0 2>&1 | tee -a "$OUT/stage2_evaluate.log"
}

case "$PHASE" in
  prepare) prepare ;;
  radar) radar ;;
  features) features ;;
  smoke) smoke ;;
  stage2) stage2 ;;
  evaluate) evaluate ;;
  all) check_pointnet; prepare; radar; features; smoke; stage2 ;;
  *) echo "Usage: bash $0 [prepare|radar|features|smoke|stage2|evaluate|all]" >&2; exit 2 ;;
esac

#!/usr/bin/env bash
set -Eeuo pipefail

# Retrain the radar path on VoD's supplied five-scan point clouds. Keep the
# existing LiDAR initialization, voxel grid, losses, splits, and schedules.
REPO=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)
BASE=${BASE:-/mnt/3D10B36523559581/Gianluca}
VOD=${VOD:-/mnt/3D10B36523559581/View-of-Delft/view_of_delft_detection_PUBLIC/view_of_delft_PUBLIC}
PY=${PY:-$BASE/Sensor-Fusion/.venv_model_v2/bin/python}
BASELINE=${BASELINE:-/media/arrubuntu20/EA0A117F0A114A49/sensor_fusion_outputs/crossmodal_fov70_refined_noaug_30ep_20261015}
META20=${META20:-/media/arrubuntu20/EA0A117F0A114A49/sensor_fusion_outputs/meta_occupancy_20261020}
OUT5=${OUT5:-/media/arrubuntu20/EA0A117F0A114A49/sensor_fusion_outputs/meta_occupancy_official5_20261021}
WORKERS=${WORKERS:-8}
RADAR_VARIANT=radar_5frames
LIDAR_ENCODER=$BASELINE/full/lidar/best_encoder.pth
NOAUG=$REPO/configs/crossmodal_no_geometric_augmentation.json
CONFIG=$REPO/configs/crossmodal_meta_occupancy.json
RAW=$OUT5/fixed_frames
RADAR_RUN=$OUT5/radar_pretrain
RADAR_ENCODER=$RADAR_RUN/best_encoder.pth
GEOMETRY=$OUT5/meta_geometry_v3
TRAIN=$OUT5/meta_train_v3
SWEEP=$OUT5/threshold_sweep_val
COMMON=$OUT5/common_support_comparison

on_error() {
  local status=$?
  printf 'Five-scan ablation stopped at line %s (exit %s); outputs remain under %s\n' \
    "$1" "$status" "$OUT5" >&2
  exit "$status"
}
trap 'on_error $LINENO' ERR

test -s "$PY"
test -s "$LIDAR_ENCODER"
test -s "$BASELINE/full/lidar/best_pretrain.ckpt"
test -s "$BASELINE/fixed_frames/val/manifest.json"
test -s "$META20/meta_train_v3/best_student.ckpt"
test -s "$META20/meta_geometry_v3/val/manifest.json"
test -s "$NOAUG"
test -s "$CONFIG"
test -d "$VOD/radar_5frames/training/velodyne"
test -s "$VOD/lidar/ImageSets/train.txt"
test -s "$VOD/lidar/ImageSets/val.txt"
if (( WORKERS < 1 )); then
  printf 'WORKERS must be positive.\n' >&2
  exit 1
fi
mkdir -p "$OUT5" "$OUT5/matplotlib" "$RADAR_RUN" "$TRAIN"
exec 9>"$OUT5/run.lock"
flock -n 9 || { printf 'Another ablation run holds %s/run.lock\n' "$OUT5" >&2; exit 1; }
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
export LD_LIBRARY_PATH="$BASE/stage2-openblas-runtime/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export OMP_NUM_THREADS=8
# The two reference entries must use the same KD-tree distances as the fast
# CPU cache builder for its exact field-by-field verification.
export CROSSMODAL_GPU_NEIGHBORS=0
export MPLCONFIGDIR="$OUT5/matplotlib" MPLBACKEND=Agg
cd "$REPO"

printf '\n=== Cache official five-scan radar and the unchanged clean LiDAR ===\n'
for split in train val; do
  "$PY" -u -m scripts.cache_crossmodal_fixed_frames \
    --vod-root "$VOD" --radar-variant "$RADAR_VARIANT" \
    --cache-root "$RAW" --split "$split" --num-workers "$WORKERS"
done

printf '\n=== Pretrain the five-scan radar encoder for 30 epochs ===\n'
radar_resume=()
if [[ -s "$RADAR_RUN/last.ckpt" ]]; then
  radar_resume=(--resume "$RADAR_RUN/last.ckpt")
elif [[ -e "$RADAR_RUN/metrics.jsonl" || -e "$RADAR_ENCODER" ]]; then
  printf 'Incomplete radar pretraining state in %s; inspect it before resuming.\n' "$RADAR_RUN" >&2
  exit 1
fi
"$PY" -u -m models.radar_lidar_crossmodal_v1.pretrain_encoder \
  --vod-root "$VOD" --radar-variant "$RADAR_VARIANT" \
  --modality radar --data-cache-root "$RAW" --output-root "$RADAR_RUN" \
  --online-augmentation-config "$NOAUG" \
  --epochs 30 --batch-size 16 --full-val-every 5 \
  --num-workers "$WORKERS" --device cuda --no-tensorboard \
  "${radar_resume[@]}"
test -s "$RADAR_ENCODER"

printf '\n=== Build two reference geometry entries per split ===\n'
for split in train val; do
  "$PY" -u -m scripts.cache_meta_occupancy \
    --vod-root "$VOD" --radar-variant "$RADAR_VARIANT" \
    --data-cache-root "$RAW" --cache-root "$GEOMETRY" \
    --radar-encoder "$RADAR_ENCODER" --lidar-encoder "$LIDAR_ENCODER" \
    --config "$CONFIG" --split "$split" --limit 2 \
    --num-workers 2 --device cuda
done
"$PY" -u -m scripts.smoke_meta_cache \
  --vod-root "$VOD" --radar-variant "$RADAR_VARIANT" \
  --data-cache-root "$RAW" --cache-root "$GEOMETRY" \
  --radar-encoder "$RADAR_ENCODER" --lidar-encoder "$LIDAR_ENCODER" \
  --config "$CONFIG" --check-gradients --device cuda

printf '\n=== Fill lossless geometry cache with %s CPU workers ===\n' "$WORKERS"
for split in train val; do
  "$PY" -u -m scripts.cache_meta_occupancy_fast \
    --vod-root "$VOD" --radar-variant "$RADAR_VARIANT" \
    --data-cache-root "$RAW" --cache-root "$GEOMETRY" \
    --radar-encoder "$RADAR_ENCODER" --lidar-encoder "$LIDAR_ENCODER" \
    --config "$CONFIG" --split "$split" --verify-existing 2 \
    --num-workers "$WORKERS"
done

printf '\n=== Joint teacher/student training for 25 epochs ===\n'
meta_resume=()
if [[ -s "$TRAIN/last.ckpt" ]]; then
  meta_resume=(--resume "$TRAIN/last.ckpt")
elif [[ -e "$TRAIN/metrics.jsonl" || -e "$TRAIN/best_student.ckpt" ]]; then
  printf 'Incomplete meta training state in %s; inspect it before resuming.\n' "$TRAIN" >&2
  exit 1
fi
"$PY" -u -m models.radar_lidar_crossmodal_v1.train_meta_occupancy \
  --vod-root "$VOD" --radar-variant "$RADAR_VARIANT" \
  --data-cache-root "$RAW" --meta-cache-root "$GEOMETRY" \
  --radar-encoder "$RADAR_ENCODER" --lidar-encoder "$LIDAR_ENCODER" \
  --config "$CONFIG" --online-augmentation-config "$NOAUG" \
  --output-root "$TRAIN" --epochs 25 --warmup-epochs 2 \
  --batch-size 8 --num-workers "$WORKERS" \
  --scan-metrics-every 5 --quick-val-frames 32 --full-val-every 5 \
  --device cuda "${meta_resume[@]}"
test -s "$TRAIN/best_student.ckpt"

printf '\n=== Sweep the five-scan model on full validation ===\n'
"$PY" -u -m scripts.sweep_meta_occupancy_thresholds \
  --vod-root "$VOD" --data-cache-root "$RAW" \
  --meta-cache-root "$GEOMETRY" \
  --checkpoint "$TRAIN/best_student.ckpt" \
  --output-root "$SWEEP" --split val \
  --batch-size 2 --num-workers "$WORKERS" --device cuda
printf '\nFive-scan native sweep complete: %s/threshold_sweep.json\n' "$SWEEP"

printf '\n=== Compare five and 20 scans on identical validation voxels ===\n'
common_args=(
  --vod-root "$VOD"
  --five-checkpoint "$TRAIN/best_student.ckpt"
  --twenty-checkpoint "$META20/meta_train_v3/best_student.ckpt"
  --five-data-cache-root "$RAW"
  --twenty-data-cache-root "$BASELINE/fixed_frames"
  --five-meta-cache-root "$GEOMETRY"
  --twenty-meta-cache-root "$META20/meta_geometry_v3"
  --split val --device cuda
)
"$PY" -u -m scripts.compare_meta_occupancy_common_support \
  "${common_args[@]}" --output-root "$COMMON/smoke" \
  --limit 2 --num-workers 0
"$PY" -u -m scripts.compare_meta_occupancy_common_support \
  "${common_args[@]}" --output-root "$COMMON/full" \
  --num-workers "$WORKERS"
printf '\nCommon-support comparison: %s/full/common_support_sweep.json\n' "$COMMON"

#!/usr/bin/env bash
set -euo pipefail

# Run in the laptop's WSL terminal. All outputs stay on the WSL filesystem.
REPO=${REPO:-/mnt/c/Users/gianl/Desktop/Thesis/radar-lidar-crossmodal}
VOD=${VOD:-/mnt/c/Users/gianl/Desktop/Thesis/View-Of-Delft dataset/view_of_delft_PUBLIC}
PRETRAIN_DATA=${PRETRAIN_DATA:-/home/gianl/stage2_conditional_local_from_scratch_20261108/fixed_frames}
GATE_SCOPE=${GATE_SCOPE:-full_grid}
RADAR_INPUT_FILTER=${RADAR_INPUT_FILTER:-0}
if [[ "$GATE_SCOPE" != full_grid && "$GATE_SCOPE" != stage2_roi ]]; then
  echo "GATE_SCOPE must be full_grid or stage2_roi" >&2
  exit 2
fi
if [[ "$RADAR_INPUT_FILTER" != 0 && "$RADAR_INPUT_FILTER" != 1 ]]; then
  echo "RADAR_INPUT_FILTER must be 0 or 1" >&2
  exit 2
fi
if [[ "$RADAR_INPUT_FILTER" == 1 && "$GATE_SCOPE" != full_grid ]]; then
  echo "Radar input filtering requires GATE_SCOPE=full_grid" >&2
  exit 2
fi
if [[ "$RADAR_INPUT_FILTER" == 1 ]]; then
  DEFAULT_OUT=/home/gianl/radar_encoder_decoder_full_grid_prefilter_gate12_20261109
  radar_filter_args=(--radar-input-filter)
else
  DEFAULT_OUT=/home/gianl/radar_encoder_decoder_${GATE_SCOPE}_gate12_20261109
  radar_filter_args=()
fi
OUT=${OUT:-$DEFAULT_OUT}
if [[ "$GATE_SCOPE" == full_grid ]]; then
  DATA=${DATA:-$OUT/full_clean_fixed_frames}
else
  DATA=${DATA:-$PRETRAIN_DATA}
fi
PY=${PY:-/home/gianl/venvs/stage2-local/bin/python}
VARIANT=${VARIANT:-radar_20frames_verified_doppler_radial}
PRETRAIN_EPOCHS=${PRETRAIN_EPOCHS:-25}
TRAIN_EPOCHS=${TRAIN_EPOCHS:-25}
PRETRAIN_BATCH=${PRETRAIN_BATCH:-8}
TRAIN_BATCH=${TRAIN_BATCH:-2}
WORKERS=${WORKERS:-2}
CACHE_WORKERS=${CACHE_WORKERS:-4}
PRETRAIN_PLOTS=${PRETRAIN_PLOTS:-1}
FULL_VAL_EVERY=${FULL_VAL_EVERY:-5}
SCAN_METRICS_EVERY=${SCAN_METRICS_EVERY:-5}
CHECKPOINT_METRIC=${CHECKPOINT_METRIC:-full_support}
PHASE=${1:-all}

cd "$REPO"
RADAR_AE_TMPDIR=${RADAR_AE_TMPDIR:-$OUT/tmp}
mkdir -p "$RADAR_AE_TMPDIR"
export PYTHONPATH="/usr/lib/python3/dist-packages:$REPO${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-1}
export CROSSMODAL_GPU_NEIGHBORS=0
export TMPDIR="$RADAR_AE_TMPDIR"
CONFIG=configs/radar_autoencoder_gate12_1p6m.json
ENCODER="$OUT/radar_pretrain/best_encoder.pth"
SUPPORT="$OUT/support12vox_1p6m"
GEOMETRY="$OUT/geometry_gate12"
MODEL="$OUT/radar_autoencoder"

pretrain() {
  local resume=()
  local plot_args=()
  if [[ "$PRETRAIN_PLOTS" == 0 ]]; then
    plot_args=(--no-plots)
  fi
  if [[ -s "$OUT/radar_pretrain/last.ckpt" ]]; then
    resume=(--resume "$OUT/radar_pretrain/last.ckpt")
  elif [[ -e "$OUT/radar_pretrain/metrics.jsonl" ]]; then
    local backup="$OUT/radar_pretrain/metrics_without_checkpoint_$(date +%Y%m%d_%H%M%S)_$$.jsonl"
    mv -- "$OUT/radar_pretrain/metrics.jsonl" "$backup"
    echo "Archived incomplete pretraining metrics: $backup"
  fi
  "$PY" -u -m models.radar_lidar_crossmodal_v1.pretrain_encoder \
    --vod-root "$VOD" --data-cache-root "$PRETRAIN_DATA" \
    --radar-variant "$VARIANT" --modality radar \
    --online-augmentation-config configs/crossmodal_no_geometric_augmentation.json \
    "${radar_filter_args[@]}" \
    --output-root "$OUT/radar_pretrain" --epochs "$PRETRAIN_EPOCHS" \
    --batch-size "$PRETRAIN_BATCH" --num-workers "$WORKERS" \
    --no-tensorboard "${plot_args[@]}" "${resume[@]}"
  test -s "$ENCODER"
}

cache() {
  test -s "$ENCODER" || { echo "Missing $ENCODER; run pretrain first" >&2; exit 1; }
  local gate_args
  if [[ "$GATE_SCOPE" == full_grid ]]; then
    gate_args=(--full-radar-gate)
  else
    gate_args=(--gate-support-root "$SUPPORT")
  fi
  for split in train val test; do
    if [[ "$GATE_SCOPE" == full_grid ]]; then
      "$PY" -u -m scripts.cache_crossmodal_fixed_frames \
        --vod-root "$VOD" --cache-root "$DATA" \
        --radar-variant "$VARIANT" --full-lidar --split "$split" \
        --num-workers "$CACHE_WORKERS"
    fi
    if [[ "$GATE_SCOPE" == stage2_roi ]]; then
      "$PY" -u -m scripts.cache_radar_pruner_stage1 \
        --vod-root "$VOD" --data-cache-root "$DATA" \
        --output-root "$SUPPORT" --radar-variant "$VARIANT" \
        --radius-m 1.6 --min-distinct-radar-voxels 12 \
        --split "$split" --num-workers "$CACHE_WORKERS"
    fi
    "$PY" -u -m scripts.cache_meta_occupancy_fast \
      --vod-root "$VOD" --data-cache-root "$DATA" \
      --cache-root "$GEOMETRY" --radar-encoder "$ENCODER" \
      "${gate_args[@]}" "${radar_filter_args[@]}" --config "$CONFIG" \
      --radar-variant "$VARIANT" --split "$split" \
      --num-workers "$CACHE_WORKERS"
  done
}

train() {
  local resume=()
  local gate_args
  if [[ "$GATE_SCOPE" == full_grid ]]; then
    gate_args=(--full-radar-gate)
  else
    gate_args=(--gate-support-root "$SUPPORT")
  fi
  if [[ -s "$MODEL/last.ckpt" ]]; then
    resume=(--resume "$MODEL/last.ckpt")
  fi
  "$PY" -u -m models.radar_lidar_crossmodal_v1.train_radar_autoencoder \
    --vod-root "$VOD" --data-cache-root "$DATA" \
    --radar-encoder "$ENCODER" --meta-cache-root "$GEOMETRY" \
    "${gate_args[@]}" "${radar_filter_args[@]}" --output-root "$MODEL" \
    --config "$CONFIG" --radar-variant "$VARIANT" \
    --epochs "$TRAIN_EPOCHS" --batch-size "$TRAIN_BATCH" \
    --num-workers "$WORKERS" --full-val-every "$FULL_VAL_EVERY" \
    --scan-metrics-every "$SCAN_METRICS_EVERY" \
    --checkpoint-metric "$CHECKPOINT_METRIC" "${resume[@]}"
}

evaluate() {
  test -s "$MODEL/best_bce_student.ckpt"
  test -s "$MODEL/best_student.ckpt"
  local name threshold checkpoint
  local thresholds=(0.01 0.025 0.05 0.1 0.15 0.2 0.25 0.3 0.35 0.4
                    0.45 0.5 0.55 0.6 0.65 0.7 0.75 0.8 0.85 0.9 0.95)
  for name in best_bce best_iou; do
    if [[ "$name" == best_bce ]]; then
      checkpoint="$MODEL/best_bce_student.ckpt"
    else
      checkpoint="$MODEL/best_student.ckpt"
    fi
    "$PY" -u -m scripts.sweep_meta_occupancy_thresholds \
      --vod-root "$VOD" --data-cache-root "$DATA" \
      --meta-cache-root "$GEOMETRY" --checkpoint "$checkpoint" \
      --split val --batch-size "$TRAIN_BATCH" --num-workers "$WORKERS" \
      --thresholds "${thresholds[@]}" \
      --output-root "$OUT/evaluation/val_$name"
  done
  read -r name threshold < <("$PY" -c '
import json, sys
reports = [(name, json.load(open(path))) for name, path in
           zip(("best_bce", "best_iou"), sys.argv[1:])]
name, report = max(reports, key=lambda pair:
                   max(row["iou_exact"] or -1 for row in pair[1]["rows"]))
print(name, report["best_thresholds"]["iou_exact"])
' "$OUT/evaluation/val_best_bce/threshold_sweep.json" \
  "$OUT/evaluation/val_best_iou/threshold_sweep.json")
  if [[ "$name" == best_bce ]]; then
    checkpoint="$MODEL/best_bce_student.ckpt"
  else
    checkpoint="$MODEL/best_student.ckpt"
  fi
  echo "Validation-selected checkpoint: $name; exact-IoU threshold: $threshold"
  ln -sfn "$checkpoint" "$OUT/evaluation/selected_student.ckpt"
  "$PY" -u -m scripts.sweep_meta_occupancy_thresholds \
    --vod-root "$VOD" --data-cache-root "$DATA" \
    --meta-cache-root "$GEOMETRY" --checkpoint "$checkpoint" \
    --split test --thresholds "$threshold" --batch-size "$TRAIN_BATCH" \
    --num-workers "$WORKERS" --output-root "$OUT/evaluation/test"
}

case "$PHASE" in
  pretrain) pretrain ;;
  cache) cache ;;
  train) train ;;
  evaluate) evaluate ;;
  all) pretrain; cache; train; evaluate ;;
  *) echo "Usage: bash $0 [pretrain|cache|train|evaluate|all]" >&2; exit 2 ;;
esac

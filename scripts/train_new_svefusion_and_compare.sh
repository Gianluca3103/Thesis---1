#!/usr/bin/env bash
# Train a new clean LiDAR+radar detector, then evaluate three LiDAR conditions.
set -euo pipefail

if (( $# < 2 || $# > 3 )); then
  echo "Usage: bash scripts/train_new_svefusion_and_compare.sh SVE_ROOT SVE_PYTHON [STAGE2_CHECKPOINT]" >&2
  exit 2
fi

REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
SVE=$(realpath "$1")
SVE_PY=$(realpath "$2")
VOD=${VOD:-/mnt/3D10B36523559581/View-of-Delft/view_of_delft_detection_PUBLIC/view_of_delft_PUBLIC}
OUT5=${OUT5:-/media/arrubuntu20/EA0A117F0A114A49/sensor_fusion_outputs/meta_occupancy_official5_20261021}
EXPORT=${EXPORT:-$OUT5/svefusion_stage2_comparison}
DATA_ROOT=${NEW_SVE_DATA_ROOT:-$OUT5/svefusion_clean_train_v1}
STAGE2_CKPT=${3:-$OUT5/stage2_sparse_diffusion_r2ldm_centroid_v8/last.ckpt}
CONFIG="$SVE/tools/cfgs/VoD_models/SVEFusion_stage2_new_clean.yaml"
TRAIN_OUTPUT="$SVE/output/VoD_models/SVEFusion_stage2_new_clean/clean_radar_bs2_80ep"

for required in "$SVE_PY" "$SVE/tools/train.py" "$SVE/tools/test.py" \
  "$SVE/tools/cfgs/VoD_models/SVEFusion.yaml" \
  "$EXPORT/export_manifest.json" "$STAGE2_CKPT"; do
  if [[ ! -s "$required" ]]; then
    echo "Missing required file: $required" >&2
    exit 1
  fi
done

export PYTHONPATH="$REPO:$SVE${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}
cd "$REPO"
"$SVE_PY" -u -m scripts.prepare_new_svefusion_detector \
  --vod-root "$VOD" --sve-root "$SVE" \
  --stage2-export-root "$EXPORT" --data-root "$DATA_ROOT" --workers 2

cd "$SVE/tools"
"$SVE_PY" -u train.py \
  --cfg_file "cfgs/VoD_models/SVEFusion_stage2_new_clean.yaml" \
  --extra_tag clean_radar_bs2_80ep \
  --batch_size 2 --workers 8 --epochs 80 \
  --val_interval 10 --ckpt_save_interval 10 --max_ckpt_save_num 10 \
  --quiet_progress

BEST_CKPT=$("$SVE_PY" "$REPO/scripts/select_svefusion_checkpoint.py" \
  --training-output "$TRAIN_OUTPUT")
export DETECTOR_DATA_ROOT="$DATA_ROOT"
bash "$REPO/scripts/run_stage2_svefusion_comparison.sh" \
  "$SVE" "$CONFIG" "$BEST_CKPT" "$SVE_PY" "$STAGE2_CKPT"

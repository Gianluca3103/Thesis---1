#!/usr/bin/env bash
# Run one frozen SVEFusion checkpoint on clean, faulty, and Stage-II-completed VoD.
set -euo pipefail

if (( $# < 4 || $# > 5 )); then
  echo "Usage: bash scripts/run_stage2_svefusion_comparison.sh SVE_ROOT TRAINED_MODEL_YAML DETECTOR_CHECKPOINT SVE_PYTHON [STAGE2_CHECKPOINT]" >&2
  exit 2
fi

REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
BASE=/mnt/3D10B36523559581/Gianluca
VOD=${VOD:-/mnt/3D10B36523559581/View-of-Delft/view_of_delft_detection_PUBLIC/view_of_delft_PUBLIC}
OUT5=${OUT5:-/media/arrubuntu20/EA0A117F0A114A49/sensor_fusion_outputs/meta_occupancy_official5_20261021}
STAGE2_PY=${STAGE2_PY:-$BASE/Sensor-Fusion/.venv_model_v2/bin/python}
S1=${S1:-$OUT5/radar_autoencoder/best_student.ckpt}
CACHE=${CACHE:-$OUT5/stage2_fixed_gate025_v1}
FOG_CACHE=${FOG_CACHE:-$OUT5/stage2_severe_fog_s5_v1}
EXPORT=${EXPORT:-$OUT5/svefusion_stage2_comparison}

SVE=$(realpath "$1")
DETECTOR_CONFIG=$(realpath "$2")
DETECTOR_CKPT=$(realpath "$3")
SVE_PY=$(realpath "$4")
STAGE2_CKPT=${5:-$OUT5/stage2_sparse_diffusion_fullregion_v9/last.ckpt}

for required in "$STAGE2_PY" "$SVE_PY" "$DETECTOR_CONFIG" "$DETECTOR_CKPT" \
  "$STAGE2_CKPT" "$S1" "$CACHE/val/manifest.json" \
  "$FOG_CACHE/val/manifest.json" "$SVE/tools/test.py" \
  "$REPO/scripts/score_stage2_svefusion.py"; do
  if [[ ! -s "$required" ]]; then
    echo "Missing required file: $required" >&2
    exit 1
  fi
done

export PYTHONPATH="$REPO:$SVE${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}
export LD_LIBRARY_PATH="$BASE/stage2-openblas-runtime/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

cd "$REPO"
"$STAGE2_PY" -u -m scripts.export_stage2_svefusion \
  --vod-root "$VOD" --stage1-checkpoint "$S1" \
  --stage2-checkpoint "$STAGE2_CKPT" \
  --fault-cache-root "$FOG_CACHE" --stage2-cache-root "$CACHE" \
  --data-cache-root "$OUT5/fixed_frames" \
  --output-root "$EXPORT" --radar-variant radar_5frames \
  --sampling-steps 64 --occupancy-threshold 0.5 --device cuda

prepare_args=(--sve-root "$SVE" --detector-config "$DETECTOR_CONFIG"
              --export-root "$EXPORT")
if [[ -n "${DETECTOR_DATA_ROOT:-}" ]]; then
  prepare_args+=(--detector-data-root "$DETECTOR_DATA_ROOT")
fi
"$SVE_PY" -m scripts.prepare_stage2_svefusion "${prepare_args[@]}"

RUN_TAG=$("$STAGE2_PY" -c 'import json,sys; print("stage2_"+json.load(open(sys.argv[1]))["stage2_sha256"][:12])' "$EXPORT/export_manifest.json")
cd "$SVE/tools"
results=()
for condition in clean faulty reconstructed; do
  "$SVE_PY" test.py \
    --cfg_file "cfgs/VoD_models/SVEFusion_stage2_${condition}.yaml" \
    --ckpt "$DETECTOR_CKPT" --batch_size 1 --workers 2 \
    --extra_tag "$RUN_TAG" --eval_tag "$condition" --save_to_file
  result_dir="$SVE/output/VoD_models/SVEFusion_stage2_${condition}/$RUN_TAG/eval"
  mapfile -t found < <(find "$result_dir" -type f -path "*/$condition/result.pkl" -print)
  if (( ${#found[@]} != 1 )); then
    echo "Expected one $condition result.pkl under $result_dir; found ${#found[@]}" >&2
    exit 1
  fi
  results+=("${found[0]}")
done

"$SVE_PY" "$REPO/scripts/score_stage2_svefusion.py" \
  --export-root "$EXPORT" \
  --clean-infos "$EXPORT/svefusion_val_infos.pkl" \
  --clean-results "${results[0]}" --faulty-results "${results[1]}" \
  --reconstructed-results "${results[2]}" \
  --output "$EXPORT/svefusion_3d_ap_r40.csv"

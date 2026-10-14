#!/usr/bin/env bash
set -Eeuo pipefail

# Complete the reusable VoD radar inputs, verify both splits, then train.
REPO=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)
BASE=${BASE:-/mnt/3D10B36523559581/Gianluca}
VOD=${VOD:-/mnt/3D10B36523559581/View-of-Delft/view_of_delft_detection_PUBLIC/view_of_delft_PUBLIC}
PY=${PY:-$BASE/Sensor-Fusion/.venv_model_v2/bin/python}
OUT=${OUT:-/media/arrubuntu20/EA0A117F0A114A49/sensor_fusion_outputs/crossmodal_native_sparse_verified_20261015_25ep}
RADAR_VARIANT=radar_20frames_verified_doppler_radial

test -s "$PY"
test -d "$VOD/lidar"
SHORT_TMP=${CROSSMODAL_TMPDIR:-/dev/shm/cm-$UID}
if (( ${#SHORT_TMP} > 60 )); then
  printf 'Temporary path is too long for PyTorch worker sockets: %s\n' "$SHORT_TMP" >&2
  exit 1
fi
mkdir -p "$SHORT_TMP" "$OUT/matplotlib"
FREE_BYTES=$(df -B1 --output=avail "$OUT" | tail -n 1 | tr -d ' ')
if (( FREE_BYTES < 15 * 1024 * 1024 * 1024 )); then
  printf 'Need at least 15 GiB free on the output volume: %s\n' "$OUT" >&2
  exit 1
fi
export TMPDIR="$SHORT_TMP" TMP="$SHORT_TMP" TEMP="$SHORT_TMP" MPLCONFIGDIR="$OUT/matplotlib"
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
export LD_LIBRARY_PATH="$BASE/stage2-openblas-runtime/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export OMP_NUM_THREADS=8 MPLBACKEND=Agg
cd "$REPO"

if "$PY" -u -m scripts.check_crossmodal_inputs \
  --vod-root "$VOD" --radar-variant "$RADAR_VARIANT"; then
  printf 'Verified Doppler radar cache already covers train and val.\n'
else
  for split in train val; do
    printf '\n=== Complete verified 20-scan radar cache: %s ===\n' "$split"
    "$PY" -u -m scripts.generate_vod_accumulated_radar \
      --vod-root "$VOD" --stack-sizes 20 --split "$split" \
      --alignment-source official-5 --output-suffix verified \
      --isolate-split-history --no-basic-validity-filter --num-workers 4

    printf '\n=== Complete Doppler radar cache: %s ===\n' "$split"
    "$PY" -u -m scripts.generate_vod_doppler_radar \
      --vod-root "$VOD" --split "$split" \
      --output-variant "$RADAR_VARIANT" \
      --frame-period-s 0.10022 --doppler-sign 1 --num-workers 4
  done

  "$PY" -u -m scripts.check_crossmodal_inputs \
    --vod-root "$VOD" --radar-variant "$RADAR_VARIANT"
fi

printf '\n=== Begin sequential encoder, teacher, and student training ===\n'
bash "$REPO/scripts/train_crossmodal_overnight.sh"

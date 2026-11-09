#!/usr/bin/env bash
set -euo pipefail

# Dedicated professor-machine run. The radar encoder is pretrained for 25
# epochs; the radar-only occupancy encoder/decoder is then trained for 100.
REPO=${REPO:-/media/arrubuntu20/EA0A117F0A114A49/Thesis-1-radar-autoencoder-fullgrid-20261109}
BASE=${BASE:-/mnt/3D10B36523559581/Gianluca}
DATA_ROOT=${DATA_ROOT:-/media/arrubuntu20/EA0A117F0A114A49/sensor_fusion_outputs}
VOD=${VOD:-/mnt/3D10B36523559581/View-of-Delft/view_of_delft_detection_PUBLIC/view_of_delft_PUBLIC}
PY=${PY:-$BASE/Sensor-Fusion/.venv_model_v2/bin/python}
PRETRAIN_DATA=${PRETRAIN_DATA:-$DATA_ROOT/crossmodal_fov70_refined_noaug_30ep_20261015/fixed_frames}
OUT=${OUT:-$DATA_ROOT/radar_encoder_decoder_fullgrid12_1p6m_100ep_20261109}
VARIANT=${VARIANT:-radar_20frames_verified_doppler_radial}

test -x "$PY" || { echo "Missing Python: $PY" >&2; exit 1; }
test -d "$VOD" || { echo "Missing VoD data: $VOD" >&2; exit 1; }
cd "$REPO"
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"

if ! "$PY" - "$PRETRAIN_DATA" "$VOD" "$VARIANT" <<'PY'
import json
import sys
from pathlib import Path
from models.radar_lidar_crossmodal_v1.fixed_frame_cache import _source_hash

root, vod, variant = Path(sys.argv[1]), str(Path(sys.argv[2]).resolve()), sys.argv[3]
for split in ("train", "val"):
    path = root / split / "manifest.json"
    if not path.is_file():
        raise SystemExit(1)
    manifest = json.loads(path.read_text())
    if (manifest.get("source_hash") != _source_hash()
            or manifest.get("vod_root") != vod
            or manifest.get("radar_variant") != variant
            or manifest.get("lidar_radar_fov_deg") != 70.0
            or manifest.get("lidar_radar_max_range_m") != 80.0):
        raise SystemExit(1)
PY
then
  PRETRAIN_DATA="$OUT/pretrain_fixed_frames"
  echo "Building compatible cropped radar pretraining cache: $PRETRAIN_DATA"
  for split in train val; do
    "$PY" -u -m scripts.cache_crossmodal_fixed_frames \
      --vod-root "$VOD" --cache-root "$PRETRAIN_DATA" \
      --radar-variant "$VARIANT" --split "$split" --num-workers 8
  done
fi

export REPO BASE VOD PY PRETRAIN_DATA OUT VARIANT
export GATE_SCOPE=full_grid PRETRAIN_EPOCHS=25 TRAIN_EPOCHS=100
export PRETRAIN_BATCH=16 TRAIN_BATCH=8 WORKERS=4 CACHE_WORKERS=8
export FULL_VAL_EVERY=1 SCAN_METRICS_EVERY=1 CHECKPOINT_METRIC=exact
exec bash "$REPO/scripts/run_radar_autoencoder_gate12_local.sh" all

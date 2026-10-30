#!/usr/bin/env bash
set -Eeuo pipefail

# Replace the professor-machine source checkout and rebuild only the caches
# needed by the 3 m / 32-slot trial. Trained checkpoints, VoD, and the working
# CUDA/MinkowskiEngine environment are deliberately kept.
REF=${REF:?Set REF to the full GitHub commit SHA to install}
[[ "$REF" =~ ^[0-9a-f]{40}$ ]] || { echo "REF must be a 40-character commit SHA" >&2; exit 1; }

BASE=/mnt/3D10B36523559581/Gianluca
REPO=$BASE/sensor-fusion-fault-localization
DATA=/media/arrubuntu20/EA0A117F0A114A49/sensor_fusion_outputs
VOD=/mnt/3D10B36523559581/View-of-Delft/view_of_delft_detection_PUBLIC/view_of_delft_PUBLIC
PY=$BASE/Sensor-Fusion/.venv_model_v2/bin/python
FOG_SIM=$BASE/Sensor-Fusion/Weather_Injector/LiDAR_fog_sim
SOURCE_RUN=$DATA/crossmodal_full_refog_20261029
SOURCE_STUDENT=$SOURCE_RUN/crossmodal/full/student/best_radar_iou.ckpt
LPCR=$SOURCE_RUN/lpcr/best.ckpt
RAW=$DATA/crossmodal_fov70_refined_noaug_30ep_20261015/fixed_frames
FOG_CACHE=$DATA/fine_association_3m32_20261030/fog_cache
VARIANT=radar_20frames_verified_doppler_radial

fail() { printf '%s\n' "$*" >&2; exit 1; }
[[ -d "$REPO" && ! -L "$REPO" && -d "$DATA" && -d "$VOD/lidar" ]] ||
  fail "Expected source checkout, output volume, or VoD LiDAR is missing"
[[ -x "$PY" && -s "$SOURCE_STUDENT" && -s "$LPCR" ]] ||
  fail "The working Python environment or required trained checkpoints are missing"
[[ -s "$FOG_SIM/fog_simulation.py" &&
   -s "$FOG_SIM/integral_lookup_tables/original/integral_0m_to_200m_stepsize_0.1m_tau_h_20ns_alpha_0.06.pickle" ]] ||
  fail "Original severity-5 fog simulator or lookup is missing"
[[ "$(realpath -e "$REPO")" == "$BASE/sensor-fusion-fault-localization" ]] ||
  fail "Checkout resolves outside its expected absolute path"
[[ "$(realpath -e "$DATA")" == "$DATA" ]] ||
  fail "Output root resolves outside its expected absolute path"

# Never erase checkpoint files accidentally placed inside the code checkout.
checkpoint_in_repo=$(find "$REPO" -type f \
  \( -name '*.ckpt' -o -name '*.pth' -o -name '*.pt' -o -name '*.safetensors' \) \
  -print -quit)
[[ -z "$checkpoint_in_repo" ]] ||
  fail "Checkpoint inside checkout must be moved before reset: $checkpoint_in_repo"

export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=1
export LD_LIBRARY_PATH="$BASE/stage2-openblas-runtime/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
"$PY" - <<'PY'
import torch, scipy, MinkowskiEngine
assert torch.cuda.is_available(), "CUDA GPU is unavailable"
print("Working environment:", torch.__version__, torch.cuda.get_device_name(0),
      "MinkowskiEngine", MinkowskiEngine.__file__, flush=True)
PY

archive=$(mktemp /tmp/thesis-reset-XXXXXXXX.tar.gz)
listing=$archive.list
trap 'rm -f -- "$archive" "$listing"' EXIT
curl -fL --retry 3 \
  "https://github.com/Gianluca3103/Thesis---1/archive/$REF.tar.gz" \
  -o "$archive"
tar -tzf "$archive" > "$listing"
grep -q '/scripts/run_fine_association_stage2.sh$' "$listing" ||
  fail "Downloaded archive lacks the 32-slot training runner"
grep -q '/configs/crossmodal_fine_association_3m32.json$' "$listing" ||
  fail "Downloaded archive lacks the 32-slot configuration"

"$PY" - "$DATA" "$SOURCE_STUDENT" "$LPCR" <<'PY'
import os
from pathlib import Path
import shutil
import sys

root = Path(sys.argv[1]).resolve(strict=True)
protected = [Path(item).resolve(strict=True) for item in sys.argv[2:]]
exact = {
    "fixed_frames", "teacher_static", "student_supervision",
    "student_supervision_compact", "stage2_gate", "stage2_gate_pilot",
    "stage2_gate_smoke", "meta_geometry_v2", "meta_geometry_v3",
    "stage2_fixed_gate025_v1", "stage2_severe_fog_s5_v1",
}
prefixes = ("stage2_gate", "stage2_fixed_", "stage2_severe_fog",
            "stage2_mild_", "meta_geometry_", "student_supervision")
checkpoint_suffixes = {".ckpt", ".pth", ".pt", ".safetensors"}

def is_cache(path: Path) -> bool:
    name = path.name
    if name in exact or name.startswith(prefixes):
        return True
    manifest = ((path / "manifest.json").is_file()
                or (path / "train" / "manifest.json").is_file())
    return manifest and ("cache" in name.lower() or name.endswith("_geometry"))

targets = []
for parent, dirs, _ in os.walk(root, topdown=True, followlinks=False):
    for name in list(dirs):
        path = Path(parent) / name
        if path.is_symlink():
            dirs.remove(name)
            continue
        if is_cache(path):
            dirs.remove(name)
            resolved = path.resolve(strict=True)
            if not resolved.is_relative_to(root) or resolved == root:
                raise RuntimeError(f"Unsafe cache path: {path}")
            if any(item.is_relative_to(resolved) for item in protected):
                raise RuntimeError(f"Cache contains a required checkpoint: {path}")
            for item in path.rglob("*"):
                if item.is_file() and item.suffix.lower() in checkpoint_suffixes:
                    raise RuntimeError(f"Cache contains a checkpoint: {item}")
            targets.append(path)

print(f"Removing {len(targets)} cache directories under {root}:", flush=True)
for path in targets:
    print("  ", path, flush=True)
for path in targets:
    shutil.rmtree(path)
PY

# The exact, preflight-checked checkout path is the only code directory removed.
rm -rf -- "$REPO"
mkdir -p -- "$REPO"
tar -xzf "$archive" --strip-components=1 -C "$REPO"
[[ -s "$REPO/configs/crossmodal_fine_association_3m32.json" ]] ||
  fail "Fresh checkout extraction failed"

cd "$REPO"
export PYTHONPATH="$REPO"
"$PY" -m scripts.smoke_stage2_sparse --device cuda

if ! "$PY" -u -m scripts.check_crossmodal_inputs \
    --vod-root "$VOD" --radar-variant "$VARIANT"; then
  for split in train val; do
    "$PY" -u -m scripts.generate_vod_accumulated_radar \
      --vod-root "$VOD" --stack-sizes 20 --split "$split" \
      --alignment-source official-5 --output-suffix verified \
      --isolate-split-history --no-basic-validity-filter --num-workers 2
    "$PY" -u -m scripts.generate_vod_doppler_radar \
      --vod-root "$VOD" --split "$split" \
      --output-variant "$VARIANT" --frame-period-s 0.10022 \
      --doppler-sign 1 --num-workers 2
  done
  "$PY" -u -m scripts.check_crossmodal_inputs \
    --vod-root "$VOD" --radar-variant "$VARIANT"
fi

for split in train val; do
  "$PY" -u -m scripts.cache_crossmodal_fixed_frames \
    --vod-root "$VOD" --cache-root "$RAW" \
    --radar-variant "$VARIANT" --split "$split" --num-workers 2
done

"$PY" -u -m scripts.cache_stage2_severe_fog \
  --vod-root "$VOD" --fog-simulator-root "$FOG_SIM" \
  --cache-root "$FOG_CACHE" --radar-variant "$VARIANT" \
  --split both --num-workers 2 --max-cache-gib 5

"$PY" -m scripts.smoke_stage2_severe_faults \
  --vod-root "$VOD" --fault-cache-root "$FOG_CACHE" --split val

printf '\nReset complete. Checkpoints and CUDA environment were kept.\n'
printf 'Code: %s\nRaw cache: %s\nFog cache: %s\n' "$REPO" "$RAW" "$FOG_CACHE"
printf 'Next: FOG_CACHE=%q bash %q\n' "$FOG_CACHE" "$REPO/scripts/run_fine_association_stage2.sh"

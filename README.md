# Thesis-1: radar–LiDAR cross-modal learning

Standalone encoder → cross-modal teacher → radar-only student pipeline,
based on the native sparse Stage-I backbone used in the earlier interpolation
views. Datasets, trained weights,
older experiments, and generated reports are excluded.

## What is here

| Path | Purpose |
| --- | --- |
| `models/radar_lidar_crossmodal_v1/` | Independent sparse-encoder pretraining, frozen-encoder teacher, radar-only student, evaluation, raw export |
| `models/radar_lidar_stage1/sparse.py` | Native Stage-I radar and LiDAR point encoders and four-scale sparse backbone |
| `models/radar_lidar_crossmodal_v1/vendor/` | Retained Downloads encoder source, inactive in this pipeline |
| `Fault_Localization_Model/vod_dataset/` | VoD decoding, calibration, verified 20-scan radar accumulation, Doppler radial compensation |
| `models/radar_lidar_stage1/` | Shared voxel, sparse, correspondence, paired-data, and augmentation utilities |
| `models/radar_lidar_stage2/` | Minimal candidate-domain contract and measured/free-space target construction for the student's confidence loss |
| `scripts/` | Verified stack generation, Doppler radial variant generation, GPU decoder smoke check |
| `configs/` | Teacher, student, and online augmentation settings |

The screenshot's Stage-I checkpoint used a native sparse backbone with 16, 32,
64 and 128 feature channels and axial support growth at level 4. A trained
128→64 adapter preserves the existing 64-channel cross-attention and student
interface. Radar uses RCS, radial and compensated Doppler velocity, and scan
age; LiDAR uses reflectivity. Both encoders are retrained independently. The
screenshot also used a surface proposal head and interpolation; the backbone
alone does not reproduce those five clouds. The student still outputs raw
fine-voxel confidence.
The student's fine-grid output is **raw confidence logits and probabilities**, with no
interpolation or spherical influence expansion. It is not a reconstructed
LiDAR point cloud or a trained Stage II diffusion model.

See [the pipeline guide](models/radar_lidar_crossmodal_v1/README.md) for
losses, metrics, checkpoint formats, and individual training commands.

## Prerequisites

Use a CUDA-capable PyTorch environment with NumPy, SciPy, tqdm, matplotlib,
and MinkowskiEngine. TensorBoard is optional with
`--no-tensorboard`. The existing professor-machine Python environment has
previously passed one-frame integration checks with the Downloads PFE;
the native Stage-I encoder architecture requires fresh GPU smoke tests.
No dataset or weights are stored in this repository.

The VoD root supplied as `--vod-root` must contain official `lidar`, `radar`,
`radar_5frames`, calibrations, labels, and `lidar/ImageSets` splits. If the
derived `radar_20frames_verified_doppler_radial` tree already exists, use it.
Otherwise, create it from the official scans and five-scan alignment:

```bash
VOD=/path/to/view_of_delft_PUBLIC
PY=/path/to/cuda/python
cd /path/to/Thesis---1
export PYTHONPATH="$PWD"

for split in train val; do
  "$PY" -m scripts.generate_vod_accumulated_radar \
    --vod-root "$VOD" --stack-sizes 20 --split "$split" \
    --alignment-source official-5 --output-suffix verified \
    --isolate-split-history --no-basic-validity-filter --num-workers 4
done

for split in train val; do
  "$PY" -m scripts.generate_vod_doppler_radar \
    --vod-root "$VOD" --split "$split" \
    --output-variant radar_20frames_verified_doppler_radial \
    --frame-period-s 0.10022 --doppler-sign 1 --num-workers 4
done

"$PY" -m scripts.smoke_crossmodal_raw_decoder
```

The 0.10022 s frame period is an explicit approximation because the stack's
seventh column records ordinal scan age. Both generators write derived files
inside the VoD root and refuse to silently replace existing outputs. Set
`--overwrite` only when intentionally rebuilding a variant.

## Training status

Phase 0 uses a masked child-distribution objective to train two independent
native Stage-I encoders and their 128→64 adapters. The loss mask now matches
the masked input's stride-8 cells. The teacher and student use only the saved
encoder weights. No trained weights are included; old Downloads-PFE teacher
and student checkpoints cannot be resumed with this encoder architecture.

## Provenance

Copied modules keep their original import paths so the pipeline can be moved
without changing checkpoint keys. The standalone `candidate_domain.py` keeps
only the immutable domain and center-coordinate API used by the current
student. Legacy interpolation and older Stage I model imports were removed
from that support module. The original repository is unchanged.

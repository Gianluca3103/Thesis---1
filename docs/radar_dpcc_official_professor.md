# Official D-PCC radar-to-radar run

This is a separate experiment from the earlier D-PCC-inspired radar model.
It fetches the [authors' D-PCC source](https://github.com/yunhe20/D-PCC) at
`d738f59284e505166d26127dbe21cc71e2fa62c9` during setup. The source is
kept outside this repository. The run uses its `AutoEncoder`, three-stage
point transformer encoder, entropy bottlenecks, density-aware sub-point
decoder, CUDA pointops/Chamfer3D, and the original SemanticKITTI training
losses and optimizer schedule. The true default `train.py` rate coefficient is
`1e-3` because that script overrides the args module default `5e-4`.

The necessary radar adaptations are recorded in `upstream_radar_patch.diff`:

- 3 input XYZ and 4 radar attributes replace 3 XYZ and 3 normal channels.
- The final decoder attribute width becomes 4, and decoded attributes are
  exposed for metrics. The geometry and density computation is untouched.
- Unused THC declarations and deprecated tensor accessor names are removed
  only to compile the authors' CUDA kernels against the professor's newer
  PyTorch. These edits do not change kernel math.
- An unused `open3d` import is removed from `models/utils.py`. That module's
  `index_points` function needs only PyTorch and einops; Open3D appears only
  in the import and is not used by training or evaluation.
- The existing ≥12 distinct radar voxels within 1.6 m filtered cache supplies
  the *whole retained frame* as both input and target. No LiDAR is loaded.
  D-PCC's SemanticKITTI code instead operates on 12 m cubes; full frames are
  required here by the experiment and may use substantially more memory.
- The original encoder reduces point count by one third at each of three
  stages and lowers the neighbor count for small frames. The adapter accepts
  the 46-return frame seen in this dataset. It requires at least 14 returns,
  the smallest count that leaves one point after all three reductions.
- Upstream `DownsampleLayer` mutates its neighbor count when processing a
  small frame. The wrapper resets it to 16 before each frame, so a sparse
  frame does not permanently reduce the receptive field of later frames.
  Within each frame the original `min(k, N)` behavior is retained.
- Some frames become empty after the ≥12/1.6 m filter. Frames with fewer than
  14 returns cannot pass through the original three-stage encoder. They are
  counted explicitly: training obtains no gradient from them; evaluation
  produces no points and includes any real target returns as missed returns.
  Reports state the number of empty and too-small frames and average model
  loss only over frames that actually ran. The smoke test chooses real,
  model-usable train and validation frames without creating points.
- Whole-grid XYZ and training-split attribute min/max are mapped to the
  authors' expected `[-1,1]` range. The inverse is used before metric scoring.
  This is a modality-specific data transform, so the run is not an identical
  reproduction of the paper's SemanticKITTI experiment.

The upstream model predicts a point set directly, with no occupancy logit.
There is therefore no legitimate occupancy threshold to sweep. Each epoch
reports its raw-output fine-grid voxel IoU, one-to-one point F1 within 0.2 m,
Chamfer squared distance, and the original loss terms. `best_voxel_iou.ckpt`
is selected for radar reconstruction quality; `best_chamfer.ckpt` follows the
paper's Chamfer-based checkpoint criterion. The first 1,296 validation frames
are **not** a small validation sample unless `--val-limit` is used for smoke.

On the professor machine, after fetching the pinned publication commit:

```bash
BASE=/mnt/3D10B36523559581/Gianluca
REPO="$BASE/sensor_fusion_outputs/Thesis-1-radar-autoencoder-prefilter-20261109"
cd "$REPO"
bash scripts/run_radar_dpcc_official_professor.sh setup
bash scripts/run_radar_dpcc_official_professor.sh smoke
bash scripts/run_radar_dpcc_official_professor.sh train
bash scripts/run_radar_dpcc_official_professor.sh evaluate
```

Or use `all` for those four phases in order. The separate output defaults to
`$BASE/sensor_fusion_outputs/sensor_fusion_outputs/radar_dpcc_official_12vox_1p6m_20261111`.
The training phase resumes from `model/last.ckpt` when present. The setup
phase needs the CUDA toolkit compiler and builds upstream custom operators;
the 1-frame smoke run verifies them before full training. Official test is
evaluated only if the matching filtered test cache already exists.

The authors [tested](https://github.com/yunhe20/D-PCC) Python 3.7,
PyTorch 1.7.1, and CUDA 11.0; the professor machine uses newer PyTorch and an
RTX 4090. The build-compatibility edits are explicit and recorded, but a
successful smoke test on that machine is still required before assuming the
upstream CUDA operators run correctly.

The professor machine currently has only the CUDA 11.8 toolkit, whereas its
PyTorch was built for CUDA 12.1. The optional
`scripts/install_cuda121_for_dpcc.sh` downloads NVIDIA's CUDA 12.1.1 runfile,
checks the MD5 published by NVIDIA, and installs **toolkit only** into
`$BASE/cuda-12.1` without changing the system driver or CUDA 11.8. It checks
for 15 GiB free on the data drive first. Run this once before `setup`; setup
then selects its `nvcc`. NVIDIA documents `--toolkitpath` and `--tmpdir` for a
custom installation location and extraction directory.

On Ubuntu 20.04, the current CompressAI wheel cannot be installed because it
requires glibc 2.34. The setup launcher therefore builds pinned CompressAI
1.2.8 from its source distribution in the existing Python environment. It
does not change the D-PCC model or its objective.

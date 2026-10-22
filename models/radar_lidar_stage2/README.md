# Stage II: radar-gated sparse residual diffusion

The LiDAR autoencoder trains from scratch on clean LiDAR. The later diffusion
phase uses the existing radar-only Stage-I checkpoint. This model adapts the
R2LDM LiDAR encoder's **downsampling sequence**
to MinkowskiEngine: `(2,2,2)`, `(2,2,2)`, `(2,1,1)`, `(2,1,1)` in `z,y,x`
order. The original R2LDM dense decoder and its checkpoints do not fit the
new grid. This implementation instead uses a sparse decoder and a sparse
latent diffusion U-Net. It is an adaptation, not an exact R2LDM reproduction.

## Data flow

1. Frozen radar-only Stage I reads calibrated VoD radar and supplies a
   fine-voxel probability map and stride-8 radar features. Its 3.2 m radar
   support is retained.
2. The probability threshold selects active cells **before** sparse decoding.
   Only these cells and their parents are allocated at the five LPCR scales.
3. One shared Minkowski LiDAR encoder processes only XYZ from clean LiDAR
   during training and faulty LiDAR during training and inference. Both map
   to the same sparse latent coordinates selected in step 2. The encoder
   accepts three channels and rejects intensity-bearing input.
4. Pretrain the LiDAR encoder and sparse LPCR decoder on clean LiDAR across
   the entire Stage-II ROI, independently of radar and Stage I. The decoder
   trains on contiguous 16 × 16 XY patches at all 16 heights, with a three-cell
   context halo. Patches cycle through all 100 XY tiles. Only core sites
   contribute to loss. All occupied core voxels and a balanced sample of empty
   core voxels supervise occupancy. Occupied cells supervise a center-relative
   XYZ offset to the
   average measured clean return in that voxel. The adapted fine-scale LPCR
   objective is `0.9 ×` occupancy BCE plus `0.1 ×` centroid offset L1.
5. Freeze the pretrained LiDAR encoder and decoder. Train a radar spatial
   adapter and sparse latent U-Net on `R = Z_clean - Z_faulty`. The U-Net
   receives a noised residual, faulty latent, aligned radar features, pooled
   Stage-I probabilities and diffusion time. Clean-latent MSE (R2LDM's
   denoising objective), residual L1, and the decoded LPCR loss update the
   U-Net and radar adapter. The residual L1 is a distinct signal from MSE.
6. Inference reads only radar and faulty LiDAR. DDIM-style sampling estimates
   the residual, `Z_clean_hat = Z_faulty + R_hat`, then LPCR predicts occupancy
   and one centroid point per predicted occupied voxel **only at selected cells**.
   By default, exported reconstruction contains only predicted voxels not
   already occupied by faulty LiDAR. Surviving faulty returns remain in the
   completed cloud instead of being duplicated as new points.
   Clean LiDAR is never used to choose coordinates or at inference. No LiDAR
   intensity is consumed or predicted.

The common region is XYZ `[0,32] × [-16,16] × [-2,2]` m with voxels
`0.2 × 0.2 × 0.25` m, giving `16 × 160 × 160` fine cells and a
`1 × 40 × 40` encoder latent grid. The vertical extent is restricted to
Stage I's `[-2,2]` m output. Each Stage-II latent cell interpolates up to four
nearby Stage-I radar feature sites within 3.2 m. The gate threshold defaults
to `0.25`; it determines a strict recall ceiling. Report `gate_coverage`
alongside Known/Full IoU. Those IoUs are measured on the Stage-II gate and
region, so they are not directly comparable with Stage-I full-grid IoU. The
output `.ply` contains new predicted points within allowed sites; the companion
`_completed.ply` contains faulty LiDAR plus those additions. Known/Full IoU in the prediction
preview scores that union. The preview also reports precision, recall, and F1
for missing occupied voxels and how many full-estimate predictions coincided
with a faulty return. Each new occupied voxel emits one point. This
preservation rule fits the synthetic deletion faults, whose surviving XYZ
returns are unchanged. For real faulty LiDAR containing spurious returns,
review the observed-voxel assumption.
The filter preserves observed fine voxels at output; it does not force the
latent residual to zero where any faulty point is present. A coarse latent
cell can contain both observed and missing fine voxels.

Synthetic diffusion training now cycles each frame through the established
severe fault plan over three epochs: `total_loss:1` (zero returns),
`fov_filter:5` (the narrowest retained FOV from 3D_Corruptions_AD), and
`fog_sim:5` (the original fog simulator with alpha 0.06 and noise 10).
Fog is applied to the full raw scan before the calibrated forward crop and
cached once. On load, only original clean XYZ returns are retained; the
simulator's shifted fog responses are dropped. The simulator can use raw
intensity internally to decide which returns are lost, but Stage II discards
all intensity before encoding. The existing losslessly compressed fog cache
can be reused without regeneration. The FOV
filter is an exact implementation of the original
severity-5 selection; total loss and FOV are computed without a data cache.
For real faulty LiDAR, supply
pre-aligned `float32` `.bin` files with four columns at
`FAULTY_ROOT/{train,val,test}/FRAME.bin` (or `FAULTY_ROOT/FRAME.bin`). The
fourth column is read for file-format compatibility and then discarded.

## Run on the professor machine

The existing CUDA environment must import PyTorch, SciPy, and MinkowskiEngine.
Use the matching official five-scan radar checkpoint and fixed-frame cache.
The Stage-II cache stores the fixed radar output and clean targets once, with
lossless compressed integer coordinates and original float precision.

```bash
cd /mnt/3D10B36523559581/Gianluca/sensor-fusion-fault-localization
export PYTHONPATH="$PWD" OMP_NUM_THREADS=8
PY=/mnt/3D10B36523559581/Gianluca/Sensor-Fusion/.venv_model_v2/bin/python
VOD=/mnt/3D10B36523559581/View-of-Delft/view_of_delft_detection_PUBLIC/view_of_delft_PUBLIC
OUT5=/media/arrubuntu20/EA0A117F0A114A49/sensor_fusion_outputs/meta_occupancy_official5_20261021
S1="$OUT5/radar_autoencoder/best_student.ckpt"
CACHE="$OUT5/stage2_fixed_gate025_v1"
FOG_CACHE="$OUT5/stage2_severe_fog_s5_v1"
FOG_SIM=/mnt/3D10B36523559581/Gianluca/Sensor-Fusion/Weather_Injector/LiDAR_fog_sim
LPCR="$OUT5/stage2_sparse_lpcr_fullregion_v9"
DIFF="$OUT5/stage2_sparse_diffusion_fullregion_v9"

"$PY" -m scripts.smoke_stage2_sparse --device cuda

# Run this block only if the fixed Stage-II geometry cache is absent.
"$PY" -u -m scripts.cache_stage2_sparse \
  --vod-root "$VOD" --data-cache-root "$OUT5/fixed_frames" \
  --stage1-checkpoint "$S1" --cache-root "$CACHE" \
  --radar-variant radar_5frames --gate-threshold 0.25 \
  --split both --num-workers 8 --device cuda

"$PY" -u -m scripts.train_stage2_sparse \
  --phase lpcr --vod-root "$VOD" \
  --data-cache-root "$OUT5/fixed_frames" --output-root "$LPCR" \
  --radar-variant radar_5frames --epochs 30 --lr 0.001 \
  --batch-size 8 --num-workers 8 --device cuda \
  --paper-metrics-every 5 --paper-metrics-frames 4

"$PY" -u -m scripts.evaluate_stage2_lpcr \
  --vod-root "$VOD" --data-cache-root "$OUT5/fixed_frames" \
  --lpcr-checkpoint "$LPCR/best.ckpt" \
  --output-root "$LPCR/paper_metrics_val" \
  --split val --limit 20 --export-pairs --device cuda

"$PY" -u -m scripts.cache_stage2_severe_fog \
  --vod-root "$VOD" --fog-simulator-root "$FOG_SIM" \
  --cache-root "$FOG_CACHE" --split both --num-workers 2

"$PY" -m scripts.smoke_stage2_severe_faults \
  --vod-root "$VOD" --data-cache-root "$OUT5/fixed_frames" \
  --fault-cache-root "$FOG_CACHE" --split val

"$PY" -u -m scripts.train_stage2_sparse \
  --phase diffusion --vod-root "$VOD" \
  --data-cache-root "$OUT5/fixed_frames" --stage2-cache-root "$CACHE" \
  --fault-cache-root "$FOG_CACHE" \
  --stage1-checkpoint "$S1" --lpcr-checkpoint "$LPCR/best.ckpt" \
  --output-root "$DIFF" --radar-variant radar_5frames \
  --gate-threshold 0.25 --epochs 50 --lr 0.0002 \
  --batch-size 16 --num-workers 8 --device cuda

"$PY" -u -m scripts.predict_stage2_sparse \
  --vod-root "$VOD" --data-cache-root "$OUT5/fixed_frames" \
  --stage2-cache-root "$CACHE" --fault-cache-root "$FOG_CACHE" \
  --stage1-checkpoint "$S1" --stage2-checkpoint "$DIFF/best.ckpt" \
  --output-root "$DIFF/val_samples" --radar-variant radar_5frames \
  --split val --limit 10 --html --device cuda
```

The last command uses a deterministic mix of the three severe faults for a
validation demonstration. Add `--fault-type total_loss`, `fov_filter`, or
`fog_sim` to preview one condition across all selected frames. Open
`$DIFF/val_samples/index.html` to inspect the results.
The four default panels show faulty LiDAR, clean LiDAR, reconstructed points
with aligned radar, and faulty LiDAR with radar and reconstruction. Each panel
can toggle any of these four layers independently. The confidence slider filters
reconstructed points down to the generation threshold set with
`--occupancy-threshold`; regenerating the preview with a lower threshold is
needed to inspect lower-confidence points. The preview reports how many clean
returns were removed by each synthetic fault.
Use `--full-estimate` to inspect the unfiltered decoder output. Use
`--zero-residual` with the same split, random seed, and occupancy threshold to
measure whether diffusion improves missing-voxel precision/recall over decoding
the faulty latent directly. These flags do not require retraining.
For real inference, add `--faulty-root "$FAULTY_ROOT"`; the clean data loader
is then disabled. A new gate threshold requires a new Stage-II geometry cache
and diffusion run; the full-region LPCR checkpoint can be reused.
The existing Stage-I, Stage-II geometry, and severe-fog caches can be reused.
LPCR and diffusion both need new v9 runs because LPCR pretraining now covers
the full ROI independently of radar.
Existing Stage-II checkpoints cannot be resumed.
Checkpoints record the Stage-I SHA-256, gate, model revision, and fault policy
to prevent incompatible runs.
The adapted LPCR loss is `0.9 L_occ + 0.1 L_offset` at fine resolution.
The diffusion loss is `L_clean_latent_MSE + L_residual_L1 + L_LPCR` with default
outer weights of one. R2LDM itself uses clean-latent MSE during diffusion;
the residual and decoded LPCR terms are this project's additions. The
30-epoch, `1e-3` AdamW/StepLR LPCR starting settings are retained.

Start with `--train-limit 2 --val-limit 2 --epochs 1 --num-workers 0`
in a new output directory to check the full Stage-I-to-Stage-II path.
The diffusion trainer can resume its own `last.ckpt` with
`--resume "$DIFF/last.ckpt"`. The fixed Stage-II cache is
required for physical batches larger than one; its compressed geometry is
decompressed in the eight data workers. Sparse coordinates carry a frame
index, so the MinkowskiEngine convolutions process all frames together without
mixing their sites. LPCR and diffusion losses retain equal per-frame weighting.

## LPCR point-cloud evaluation

The autoencoder reports full-ROI squared-L2 Chamfer distance, squared-L2
Hausdorff distance, precision, recall, and F-score at 5 cm every five epochs
on four fixed validation frames. These use the geometric definitions in
R2LDM's released evaluator. The trainer still selects `best.ckpt` by validation
loss; `scripts.evaluate_stage2_lpcr` scores that checkpoint on more frames.
Its optional `gen/` and `gt/` XYZ files can be passed to R2LDM's evaluator for
BEV JSD/MMD and optional PointNet FPD. Use this project's ROI
`0 -16 -2 32 16 2` and metric voxel size `0.125 0.125 0.125` there.

```bash
cd /path/to/R2LDM
"$PY" eval_metric_2025.py \
  --data-root "$LPCR/paper_metrics_val" \
  --point-cloud-range 0 -16 -2 32 16 2 \
  --voxel-size 0.125 0.125 0.125 \
  --fscore-threshold 0.05 --device cuda \
  --output "$LPCR/paper_metrics_val/r2ldm_summary.json"
```

Add `--pointnet-fpd` only when the pretrained PointNet extractor is available.

These scores are autoencoder reconstructions from **clean LiDAR input**.
R2LDM's published tables evaluate the complete radar-to-point-cloud model on
different preprocessing grids; direct numerical comparison is not valid.
Known/Full IoU remains a gate-specific metric for the later diffusion phase.

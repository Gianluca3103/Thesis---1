# Stage II: radar-gated sparse residual diffusion

The LiDAR autoencoder trains from scratch on clean LiDAR. The later diffusion
phase uses the existing radar-only Stage-I checkpoint. This model adapts the
R2LDM LiDAR encoder's **downsampling sequence**
to MinkowskiEngine: `(2,2,2)`, `(2,2,2)`, `(2,1,1)`, `(2,1,1)` in `z,y,x`
order. The original R2LDM dense decoder and its checkpoints do not fit the
new grid. This implementation instead uses a sparse decoder and a sparse
latent diffusion U-Net. It is an adaptation, not an exact R2LDM reproduction.

## Geometry-only mild-fault baseline: full eligible validation

Set `FULL_VAL=1` in `scripts/run_stage2_geometry_epsilon_ray_offset_mild.sh`.
The trainer then evaluates the loss on **all 1,296 validation frames** each
epoch and samples the diffusion model on each of those frames. Sampling uses
64 steps and a 0.5 occupied-cell threshold by default (`FULL_VAL_STEPS` and
`FULL_VAL_THRESHOLD` override them). This is substantially slower than the
former 64-frame validation pass. A fixed seed per frame makes the sampled
occupancy comparison repeatable across epochs.

The prediction set is **added occupied voxels inside the candidate gate**:
at least three distinct radar voxels within 1.6 m, inside the cached synthetic
injector footprint, excluding cells observed by faulty LiDAR. The target set
is clean occupied voxels inside that same gate. Therefore the scores measure
reconstruction occupancy on eligible cells only. They do not score the entire
scene, preserved faulty points, or any predicted centroid offsets.

`eligible_iou` is exact voxel intersection over union. `eligible_iou_0p2m`
counts a maximum one-to-one match between predicted and target **voxel
centers** within 0.2 m, then uses matches / (predicted + target - matches).
The 0.2 m rule allows a horizontal neighbor on the 0.2 m grid, but not a
vertical neighbor 0.25 m away. The scores are pooled over all frames and also
reported by fault. The epoch report is written atomically to
`$OUT/eligible_validation/epoch_NNN.json` and summarized in `train.log`.
If a run stops after saving `last.ckpt` but before finishing sampled
validation, resuming completes that epoch's report first. Resuming from the
old 64-frame protocol resets the best-loss comparison; subsequent
`best.ckpt` selections use full validation loss.

For qualitative geometry-baseline previews,
`scripts.predict_stage2_sparse --html --per-fault 2 --skip-first 64`
selects two validation examples from each cached mild-fault type after the
first 64 frames. The viewer labels the actual cached
fault, shows eligible voxel centers, and reports occupancy-only eligible IoU
and IoU@0.2m for each sample alongside the reconstructed point cloud.
After the 25-epoch batch-eight run, `scripts/run_stage2_geometry_post25.sh`
checks that `last.ckpt` is epoch 25, sweeps 512 other validation frames
(256 calibration and 256 held-out) using `best.ckpt`, writes eight balanced
interactive previews, and scores the selected `best.ckpt` on all 1,296
eligible validation frames. It performs no additional training updates.
When the interactive HTML viewer cannot open, run
`scripts.render_stage2_sample_png --sample-root "$OUT/val_samples_best_f1"`.
It reads the already exported NPZ arrays and writes a `static_png` directory
with eight top/side-view PNGs and a `contact_sheet.png`. These open directly
in VS Code's image preview; rendering requires only NumPy and the Python
standard library, with no Firefox, display server, or model inference.
For interactive rotation on a slower desktop, run
`scripts.make_stage2_lite_viewer --input-root "$OUT/val_samples_best_f1" --output-root "$OUT/val_samples_best_f1_lite"`.
This reuses the already generated frame HTML, caps only the points sent to the
browser (6,000 per cloud and 4,000 eligible-cell centers by default), and
keeps the full counts and metrics in the viewer. The original PLY and NPZ
files remain untouched. Open the light directory's `index.html` to rotate
and compare the sample clouds.

## Data flow

1. Frozen radar-only Stage I reads calibrated VoD radar and supplies a
   fine-voxel probability map and stride-8 radar features. Its 3.2 m radar
   support is retained.
   Stage I can be the direct radar occupancy autoencoder or the trained
   cross-modal raw-confidence student. The latter uses the teacher only during
   its own training; Stage-II inference still reads radar and faulty LiDAR.
   Changing Stage-I checkpoints requires rebuilding the Stage-II gate cache
   and retraining diffusion, since both are bound to the Stage-I weight hash.
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

For a separate mild-fault dataset, run `python -m scripts.cache_stage2_mild_faults`
with `--vod-root`, `--data-cache-root`, `--fog-cache-root`, and `--output-root`.
It assigns one deterministic, balanced fault to each train/validation frame:
severity-5 fog or a spatially contiguous azimuth loss removing 30%, 40%, or
50% of clean returns. Partial-loss frames retain at least half their original
returns; severity-5 fog can retain less. Fog uses the existing simulator cache
and original-return filter, so no synthetic fog points enter the output.
The dataset stores only XYZ as losslessly compressed `.npz` files, plus a
manifest describing each frame's fault. Pass the output directory to
`--faulty-root` when training Stage-II diffusion on this dataset.

To inspect the cached mild inputs, run:

```bash
python -m scripts.visualize_stage2_mild_faults \
  --mild-root /path/to/mild_cache \
  --output-root /path/to/mild_views \
  --split val --per-fault 2
```

Open `mild_views/index.html` in a browser. Each example has synchronized 3D
views of the clean reference, faulty input, missing points, and their overlay.
Full-resolution PLY files accompany the browser views. Use `--search-root`
instead of `--mild-root` to find a single mild cache within an output directory;
use repeated `--frame-id` options to inspect specific frames. The viewer does
not run Stage I or Stage II predictions.

To compare a lighter fog input against the current severity-5 input on the
same validation frames, run `python -m scripts.preview_stage2_milder_fog` with
`--search-root` (or `--mild-root`), `--fog-simulator-root`, and `--output-root`.
The default lighter setting is severity 2 (alpha 0.01); `--severity 3` uses
alpha 0.02. The script applies the original fog simulator to raw VoD LiDAR,
crops to the Stage-II radar sector, filters to original detectable returns,
and writes interactive HTML and full-resolution PLY files. It creates a
preview for inspection, not a replacement training cache.

For a separate training-ready severity-2 mixed-fault cache and a fresh
hierarchical Stage-I model, run
`bash scripts/run_hierarchical_stage1_mild_fog2.sh` on the professor machine.
It builds balanced train/validation assignments of severity-2 fog and 30/40/50%
contiguous partial loss using the 20-frame radar fixed cache, verifies every
saved entry, writes interactive examples, trains Stage I from scratch, and
evaluates it on validation frames excluded from checkpoint selection. If the
previous hierarchical checkpoint is present, it is evaluated on the same new
fault set for a matched comparison. The new Stage-I training mode reads the
saved faulty XYZ and targets clean fine voxels absent from that input.
After training, `bash scripts/run_hierarchical_mild_fog2_threshold_sweep.sh`
compares old and new Stage I on the same 256 held-out validation frames (about
20% of the validation split), using a separate 256-frame calibration subset.
It sweeps score thresholds at 0.0001 intervals over the full candidate pool,
both for the full ROI and for the 1 m radar eligibility rules requiring 3 or
5 raw returns. Reports include maximum recall at 20% precision and maximum
precision at 90% and 95% recall, with calibrated and descriptive held-out
operating points clearly separated.

## Run on the professor machine

For a new official five-scan radar-only Stage-I and selective-addition Stage-II
experiment, run `bash scripts/run_fresh_radar_stage2_selective.sh` from the
repository root. It builds a new severity-5 fog cache, checks that the fault
injector removes returns, pretrains the radar encoder, trains Stage I, builds
the per-frame 0.25 probability gate, trains full-region LPCR, and trains the
selective-addition diffusion model. The default output is
`radar_stage2_selective_20261028` under `sensor_fusion_outputs`; override `OUT`
for another fresh run. `RAW`, `FOG_SIM`, `PY`, `VOD`, and `LIDAR_ENCODER` can be
overridden as environment variables. Re-running the script with the same `OUT`
resumes its checkpoints. The earlier LiDAR encoder is used only to verify the
spatial geometry cache; its features are not inputs to the radar-only Stage I.

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
Vertical drags pitch and horizontal drags yaw; the default drag mode locks to
the dominant axis. Select Pitch only or use the Pitch angle slider to adjust
elevation without changing yaw. Shift-drag or right-drag pans. The camera speed
slider defaults to 0.7× and can be reduced to 0.2× for precise movement.
To update already generated HTML viewers without rerunning diffusion, run
`"$PY" -m scripts.refresh_stage2_viewer --output-root "$DIFF/val_samples"`
and reopen the same `index.html`.
Use `--full-estimate` to inspect the unfiltered decoder output. Use
`--zero-residual` with the same split, random seed, and occupancy threshold to
measure whether diffusion improves missing-voxel precision/recall over decoding
the faulty latent directly. These flags do not require retraining.
The prediction script accepts an existing v8 diffusion checkpoint for this
comparison and retains that checkpoint's original Stage-I probability input
to LPCR. The v8 checkpoint still cannot resume v9 full-region training.
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
The diffusion loss is `L_clean_latent_MSE + L_residual_L1 + L_LPCR +
L_addition + L_healthy` with default outer weights of one. R2LDM itself uses
clean-latent MSE during diffusion; the other terms are this project's additions.
The 30-epoch, `1e-3` AdamW/StepLR LPCR starting settings are retained.

### Fixed-geometry epsilon/ray/offset baseline

`bash scripts/run_stage2_geometry_epsilon_ray_offset_mild.sh` trains Stage II
without a learned Stage I checkpoint. It reuses the existing 1.6 m support
cache: a candidate must have at least three **distinct radar voxels** within
1.6 m of its center. A second cache records where the synthetic fault injector
acted. For partial-loss faults, this is its removed azimuth sector across the
ROI, including a one-voxel boundary halo; for fog, the full scan was exposed.
The gate excludes the larger untouched, reliable sector and voxels already
occupied by the faulty LiDAR. All remaining cells are scored; no learned gate
threshold prunes them. The trainable radar adapter
receives deterministic pooled radar count, RCS, Doppler, age, and position
statistics, not a learned Stage I feature map. The footprint is an **oracle
synthetic-fault input** cached from the injector's seed and clean input. It is
unavailable for a real unknown fault without a separate fault detector.

This run reuses the frozen full-region LPCR LiDAR encoder and decoder, and
trains the Stage-II radar adapter, sparse epsilon U-Net and addition head on
the cached mild-fog2/partial-loss data. The **only** optimized terms are
`MSE(predicted_noise, sampled_noise)`, a clean-centroid XYZ offset L1 inside
measured clean voxels, the first-hit ray term, and the extra-return ray term.
The latter three are multiplied by the existing bounded epsilon task weight.
No healthy-region oracle, addition-classification loss, LPCR occupancy BCE,
or direct residual loss is called in this mode. Clean LiDAR supplies training
targets and rays only, never inference inputs. The ray labels use direction
matching as a proxy because exact LiDAR firing IDs are unavailable.

The launcher uses the existing 20-frame radar variant, 1.6 m support cache,
injector-region cache, and 64 validation frames for checkpoint selection. It
then evaluates 512
different validation frames, alternating 256 calibration and 256 held-out
frames; the first 64 are excluded. The evaluation samples with 64 diffusion
steps and reports the geometry ceiling, keep-all precision, and held-out
metrics at thresholds selected on calibration. `EVAL=0` skips this expensive
second pass. The launcher now uses eight frames per GPU optimizer step.
Losses are averaged per frame; the learning rate remains `2e-4`. This gives
roughly eight times fewer optimizer updates per epoch than the batch-one run,
so compare validation outcomes rather than epoch numbers alone. An eight-frame
GPU smoke run can use a **new** output directory:

```bash
OUT=/media/arrubuntu20/EA0A117F0A114A49/sensor_fusion_outputs/stage2_geometry1p6m_injector_region_batch8_smoke \
BATCH_SIZE=8 EPOCHS=1 TRAIN_LIMIT=32 VAL_LIMIT=8 EVAL=0 \
bash scripts/run_stage2_geometry_epsilon_ray_offset_mild.sh
```

The default full run writes a separate checkpoint and validation report under
`stage2_geometry1p6m_injector_region_epsilon_ray_offset_mild_batch8_20261107`.
`RAY_WORKERS=8` lets SciPy match fixed clean-return rays on eight CPU threads
instead of one. This changes execution speed, not the ray labels or objective.
The fixed hit, front-of-hit, and extra-return indices can also be cached once
with `scripts.cache_stage2_ray_labels` and selected with `RAY_LABELS=/path/to/cache`
when launching training. Logits and ray-loss gradients still run on the GPU.
This optional cache is validated against the ordered candidate coordinates,
fault cache, support mask, injector footprint, and ray tolerances. It can be
enabled when resuming an existing batch-eight checkpoint because it does not
change the objective or checkpoint identity.

```bash
python -m scripts.cache_stage2_ray_labels \
  --vod-root "$VOD" --data-cache-root "$RAW" --faulty-root "$MILD" \
  --support-cache-root "$SUPPORT" --fault-region-root "$REGIONS" \
  --output-root "$RAY_LABELS" --split both --device cuda
RAY_LABELS="$RAY_LABELS" bash scripts/run_stage2_geometry_epsilon_ray_offset_mild.sh
```

The launcher resumes from `last.ckpt` in the same output directory. Batch size
is bound to the checkpoint, so use a separate output directory for batch eight.
Build the footprint sidecar with `scripts.cache_stage2_fault_regions` after
the fixed-frame and mild-fault caches; it verifies the partial-fault outputs
against the deterministic injector. GPU memory and speed
must be verified by the smoke run: the full rule often exposes about 120,000
fine candidates per frame, substantially more than an old learned gate.

```bash
python -m scripts.cache_stage2_fault_regions \
  --vod-root "$VOD" --data-cache-root "$RAW" \
  --faulty-root "$MILD" --output-root "$REGIONS" \
  --radar-variant radar_20frames_verified_doppler_radial --split both
```

### Separate epsilon-prediction ablation

`bash scripts/train_stage2_epsilon_ablation.sh` reuses the radar Stage-I
checkpoint, per-frame 0.25 gate cache, severity-5 fault cache, and full-region
LPCR checkpoint from `radar_stage2_ray0_20261028`. It saves a separate
`radar_stage2_epsilon_ray025_20261029` checkpoint; the residual run and its
loader behavior remain unchanged. Set `SOURCE_RUN` and `OUT` to override these
defaults. Both modes emit at most one reconstructed centroid per allowed voxel.
The common LPCR emission path enforces this for residual, epsilon, and legacy
checkpoints: if a caller supplies duplicate selected cells, it keeps the
highest-probability point, and offsets stay strictly inside their source cell.
Original faulty LiDAR returns are preserved and are not subject to this
generated-point limit.

For the epsilon mode, the sparse U-Net predicts the sampled Gaussian noise
`epsilon_hat` instead of the clean-minus-faulty latent residual. The bounded
residual estimate is recovered as
`R_hat = clip((R_t - sqrt(1-alpha_bar_t)*epsilon_hat)/sqrt(alpha_bar_t), -2, 2)`;
the shared LiDAR encoder's tanh output bounds the true residual to this range.
Training uses `MSE(epsilon_hat, epsilon)` plus the LPCR, addition, healthy,
and ray losses on `Z_F + R_hat`. Decoded losses are multiplied by
`min(1, sqrt(alpha_bar_t/(1-alpha_bar_t)))` to control gradients at the noisiest
steps. One timestep is sampled per epsilon batch so this weight applies to
every frame in that batch. Direct residual L1 defaults to zero for this run.
The launch script keeps the two ray weights at 0.25 and verifies the epsilon
forward, backward, and one-point-per-voxel sampler before training. Epsilon
checkpoints can be used with `scripts.predict_stage2_sparse` and the existing
evaluation/export tools through `load_stage2`.

Two optional ray terms are implemented for later ablations. `L_hit` rewards an
addition at a missing clean return and penalizes earlier unobserved candidate
cells on its matched ray; `L_extra` penalizes an addition clearly behind the
first clean return on a matched ray. Both operate on addition logits within
the radar-selected gate. They do not label unmatched cells as free or require
clean LiDAR at inference. VoD point files lack beam IDs, so ray matching uses
XYZ direction (default 1.5 degrees), lateral distance (0.15 m), and a
half-voxel-diagonal range margin. These are tunable geometric proxies.
`L_hit` partly overlaps the existing missing-positive and known-free losses.
The trainer defaults to `--ray-hit-weight 0 --ray-extra-weight 0`, which skips
ray matching entirely and preserves the current objective. A future ablation
can set either weight above zero in a separate training run. The new flags
and geometry settings are recorded in diffusion checkpoints.

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

## Stage-I and Stage-II completion threshold sweep

`scripts.sweep_stage1_stage2_completion` scores the radar-only Stage-I occupancy
map and the Stage-II **completed cloud** (surviving faulty LiDAR plus generated
points) against clean LiDAR. It uses the same deterministic severe fault plan as
diffusion training and the detector export. Stage I and its faulty baseline are
restricted to radar support within the Stage-II ROI; Stage II and its faulty
baseline use the whole Stage-II ROI of the paired, radar-FOV-cropped LiDAR scan. This makes each within-stage improvement
comparable, while the two stages have explicitly different evaluation scopes.

The sweep reports exact occupied-voxel IoU/F1, 0.2-m-tolerant IoU/F1 using
one-to-one matching of voxel centers, and symmetric squared-L2 Chamfer after
0.125-m centroid preprocessing (units m²; lower is better). The faulty-baseline gains are reported for all
four overlap metrics; Chamfer reduction uses only frames on which both faulty
and predicted clouds have a defined Chamfer value. In particular, the
total-loss faulty baseline has undefined Chamfer rather than an invented large
value. Scores are pooled over voxels for IoU/F1 and averaged per valid frame
for Chamfer. Results are saved overall, by fault, and per threshold.

Stage I probability and Stage II decoder occupancy thresholds are swept
independently, with exact IoU as the default selection criterion. The Stage-II
radar gate remains at the value recorded in its checkpoint (usually 0.25):
changing the gate would require rebuilding its geometry and retraining the
diffusion run. Stage-II diffusion is sampled once per frame at the lowest
decoder threshold, then the same decoded points are filtered for higher
thresholds. Each frame's metrics are saved for interruption-safe resume.
Selecting a threshold and reporting scores on the same validation split is a
validation analysis; an unbiased final estimate needs a separate held-out set.

```bash
"$PY" -u -m scripts.sweep_stage1_stage2_completion \
  --vod-root "$VOD" --data-cache-root "$OUT5/fixed_frames" \
  --stage2-cache-root "$OUT5/stage2_fixed_gate025_v1" \
  --fault-cache-root "$OUT5/stage2_severe_fog_s5_v1" \
  --stage1-checkpoint "$OUT5/radar_autoencoder/best_student.ckpt" \
  --stage2-checkpoint "$OUT5/stage2_sparse_diffusion_r2ldm_centroid_v8/last.ckpt" \
  --output-root "$OUT5/stage1_stage2_completion_sweep_v1" \
  --split val --device cuda
```

Use `--limit 2` and a separate output directory for a quick integration run.
The full run writes `threshold_metrics.json` and `threshold_metrics.csv`.

## Frozen SVEFusion detector comparison

`scripts/run_stage2_svefusion_comparison.sh` exports every official VoD
validation frame and runs **one existing, clean-trained SVEFusion checkpoint**
three times: original clean LiDAR plus radar, severe faulty LiDAR plus radar,
and faulty LiDAR with Stage-II additions plus radar. The exporter applies the
same deterministic severe fault protocol, radar-conditioned Stage-I gate, and
missing-voxel Stage-II sampler as `predict_stage2_sparse.py`. Its clean scan is
used to synthesize the fault and label the reference condition; it is never
passed to the Stage-I or Stage-II prediction call. Total loss deletes the
full LiDAR scan; severe FoV loss filters the full scan. The available severe
fog cache covers the Stage-II radar sector, so original points outside that
sector remain for the fog subset.
Surviving original returns retain their original intensity. Stage II predicts
XYZ only, so generated points receive intensity zero. The detector's radar
files, calibration, model weights, voxelization, FoV filtering, labels, and
validation IDs are identical in every condition. The script refuses incomplete
exports and mismatched validation frame orders.

On the professor machine, locate the trained **SVEFusion LiDAR+radar**
checkpoint and its exact training model YAML (both are required). The latter
must be the YAML used for that checkpoint, not a guessed baseline config.
Then run from the updated reconstruction repository:

```bash
cd /mnt/3D10B36523559581/Gianluca/sensor-fusion-fault-localization
SVE=/absolute/path/to/SVEFusion
DETECTOR_CONFIG="$SVE/tools/cfgs/VoD_models/YOUR_TRAINED_CONFIG.yaml"
DETECTOR_CKPT=/absolute/path/to/checkpoint_epoch_N.pth
SVE_PY=/absolute/path/to/svefusion/environment/bin/python
STAGE2=/media/arrubuntu20/EA0A117F0A114A49/sensor_fusion_outputs/meta_occupancy_official5_20261021/stage2_sparse_diffusion_fullregion_v9/last.ckpt
bash scripts/run_stage2_svefusion_comparison.sh \
  "$SVE" "$DETECTOR_CONFIG" "$DETECTOR_CKPT" "$SVE_PY" "$STAGE2"
```

If the detector training YAML has a placeholder or moved `DATA_PATH`, set
`DETECTOR_DATA_ROOT` to its existing SVEFusion VoD data root before running.
When that old data root is unavailable, preparation rebuilds only the clean
validation reference under the comparison output: symlinks to original VoD
sensor files, normalized KITTI labels matching the detector's training export,
and `vod_infos_val.pkl`. It does not retrain the detector or copy point clouds.
The runner uses the established model-v2 environment for Stage II. It resumes
an interrupted point export, then writes
`$OUT5/svefusion_stage2_comparison/svefusion_3d_ap_r40.csv`. The CSV has
KITTI 3D AP_R40 for Car, Pedestrian, and Cyclist on all validation frames
and separately for total-loss, severe-FoV, and severe-fog frames. These are
full detector FoV scores; they include objects outside the Stage-II region.

To train a **new** SVEFusion detector from clean VoD LiDAR plus unmodified
five-frame radar, then run the same three-condition comparison, use
`scripts/train_new_svefusion_and_compare.sh SVE_ROOT SVE_PYTHON`. It creates a
small clean training reference under `$OUT5/svefusion_clean_train_v1` with
symlinked raw sensor data and normalized labels, and generates only the
train/validation info pickles. Its model config is based on SVEFusion's own
architecture and optimizer settings; the runner uses batch size 2, 80 epochs,
validation every 10 epochs, and selects the saved epoch with the highest clean
validation moderate 3D mAP_R40. The detector starts from random weights in a
new output directory and automatically resumes its own checkpoints if the
training run is interrupted. The existing Stage-II export is reused.

### Conditional reconstruction without clean LiDAR at prediction time

`scripts/run_stage2_conditional_faulty_start.sh` is a separate 125-epoch
faulty-start experiment. It reuses the frozen LiDAR encoder and LPCR decoder,
but randomly initializes the conditional sparse U-Net and radar adapter. On
the same ≥12-radar-voxel/1.6 m gate and mild-fault data, it encodes the faulty
LiDAR as the starting latent, adds Gaussian noise at fixed timestep 499, and
trains against the clean-LiDAR latent. The MSE target is the noise-equivalent
correction `(x_t - sqrt(alpha_t) * clean_latent) / sqrt(1-alpha_t)`; it is not
pure Gaussian epsilon because it also corrects the fault. Validation and test
use only faulty LiDAR, radar, and deterministic noise for a one-step clean
latent estimate, then load clean LiDAR solely for metrics. All 1,296
validation frames are evaluated every epoch. The run uses its own output
directory and writes a test report and, if the Gaussian-start report exists,
`comparison_with_gaussian_start.json`. The two runs share data, gate, frozen
LPCR, and evaluation metrics, but use different starting states and samplers;
compare the final validation/test IoU rather than their training MSE values.
Each epoch logs the sole optimized loss (noise-equivalent MSE) for training
and validation. Validation uses a fixed noise draw for a stable loss curve.
It also logs full-scene exact IoU and 0.2 m voxel-center occupancy IoU; both
use a fixed 0.5 *occupancy probability* cutoff. The checkpoint selection
remains based on full-scene exact IoU at that cutoff.

`scripts/run_stage2_conditional_from_scratch.sh` trains a fresh conditional
epsilon-prediction sparse U-Net and radar adapter from random initialization.
The pretrained LiDAR encoder and LPCR decoder remain frozen. Training uses
clean LiDAR only to form a noisy target latent and its epsilon target;
faulty LiDAR and measured radar condition the U-Net. No previous denoiser
checkpoint is loaded. Every epoch validates all 1,296 frames by starting
from Gaussian noise and running 64 DDIM steps with only faulty LiDAR and
radar entering the prediction path. The best checkpoint maximizes full-scene
exact IoU at a fixed 0.5 threshold. After 125 epochs by default, the runner
calibrates a full-scene threshold on validation and scores all official test
frames. This is a long GPU run because validation performs a full reverse
chain every epoch; `EPOCHS` and `VAL_STEPS` can be overridden explicitly.
The prior BCE addition head is excluded from this run because it was trained
with clean-seeded decoder features; it needs its own generated-feature trial.

Set `FAULT_REGIONS` to a distinct injector-footprint cache root when running
`scripts/run_stage2_conditional_from_scratch.sh` to train the same clean-start
objective inside the synthetic fault-area gate. The launcher builds train,
validation, and test footprints before training and uses a separate output
folder. For partial-loss faults, reliable azimuth sectors are excluded before
constructing the sparse hierarchy; fog remains full-scan. The clean LiDAR
latent supplies the training noise target, while validation/test generation
starts from Gaussian noise with faulty LiDAR and radar conditioning. The
injector footprint is an oracle diagnostic input for these synthetic faults.

For a valid evaluation of the older conditioned denoiser without retraining,
`scripts/run_stage2_conditional_reconstruction.sh` evaluates the saved
faulty-LiDAR/radar-conditioned epsilon denoiser as a conditional generator.
The model receives only the measured radar, cached faulty LiDAR, the fixed
12-radar-voxel/1.6 m coordinate rule, and frame-seeded Gaussian noise.
It performs 64 DDIM reverse steps by default and decodes the resulting
latent with frozen LPCR. A voxel already observed by faulty LiDAR is kept;
decoder predictions can only add unobserved voxels inside the radar gate.

Validation and official test prediction rows contain no clean LiDAR. The
evaluator loads clean LiDAR only after producing each batch of predictions,
for threshold calibration and metrics. It selects the addition threshold by
full-scene exact IoU on validation calibration blocks, includes a no-addition
option, and reports before/after exact and 0.2 m occupancy metrics on the
remaining validation blocks and official test. The checkpoint was originally
trained with epsilon MSE on noisy clean latents conditioned on faulty LiDAR
and radar, so this evaluates whether that training actually supports
generation from noise. No clean latent seeds the reverse chain.

The existing addition head was trained on clean-seeded decoder features. It
is deliberately excluded from this deployment-style evaluation because its
features shift when the denoiser starts from Gaussian noise. A new head must
be trained on generated features before its BCE and healthy-region penalty
can be judged in the deployable pipeline.

### Earlier clean-seeded addition-head ablation

`scripts/run_stage2_reliable_addition_mild.sh` starts a separate experiment
from the saved one-step, faulty-LiDAR/radar-conditioned denoiser. The denoiser,
radar adapter, and LPCR decoder are frozen. Only the existing Stage-II
`AdditionHead` is trained. It decides whether to add a voxel that the faulty
LiDAR did not already observe, using decoder features and local faulty-LiDAR
density. Observed faulty voxels are always retained.

The head loss is the ordinary mean binary cross-entropy over all unobserved
gated candidates, plus one extra negative BCE on empty candidates in locally
well-retained regions (`healthy_region_negatives`). The standard BCE includes
both missing clean cells and cells with no clean-LiDAR return; these are not
separate balanced terms. A missing return is not proof of physical free space.
Clean LiDAR labels training examples but is not given to the head as a feature.
The radar gate alone can include untouched regions; there is no
injector-footprint term or inference mask. The denoiser's epsilon-MSE objective
is unchanged and its weights are frozen during head training. This remains an
oracle-assisted diagnostic because the denoiser starts from a noisy
**clean-LiDAR** latent, not from faulty LiDAR alone.

Validation each epoch reports exact full-ROI IoU at threshold 0.5 against the
faulty baseline. The post-training evaluator sweeps 0.000–1.000 on validation,
includes an explicit no-additions option, selects the threshold by full-ROI
exact IoU on calibration frames, and reports exact and 0.2 m occupancy IoU/F1
for held-out validation and the official test set. Failure to improve full-ROI
IoU is visible.

### Oracle fault-area gate diagnostic

For partial-loss faults, the active cells are the intersection of the
12-radar-voxel/1.6 m support and the injector's removed azimuth sector. A
conservative inward voxel margin excludes cells that could touch the reliable
sector. Fog was applied to the full scan, so its gate is unchanged. The
faulty LiDAR's observed voxels remain in the completed output and are never
emitted as additions. Training and inference use the same filtered support.
The denoiser sees clean LiDAR only as a training target, while the footprint
cache builder uses the synthetic injector's clean input and seed to recover
the known fault sector.

The injector footprint is an **oracle synthetic-fault sidecar**, not an input
available for unknown real faults. These results diagnose the reconstruction
model's quality when the fault location is supplied. The evaluator reports
both full-scene faulty-versus-completed IoU/F1 and addition-only scores inside
the active gate, with exact and 0.2 m occupancy matching. It also reports
mean active cells to verify how much of the radar gate was excluded.

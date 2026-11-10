# Filtered radar-to-radar autoencoder

This is a separate experiment from `train_radar_autoencoder.py`. That earlier
script predicts **LiDAR occupancy** from radar. This experiment uses only radar:
its input and target are the identical retained radar returns from each VoD
frame. No LiDAR data, labels, or LiDAR checkpoint enter the model or loss.

## Data and model

- Work inside the existing ±51.2 m horizontal, ±2 m vertical encoder grid.
- Evaluate the ≥12 distinct radar voxels within 1.6 m rule once on the original
  radar frame. Keep every return in a qualifying voxel, including its seven
  measured columns. An isolated return is removed from both input and target.
- Cache the filtered returns separately under `filtered_radar/{train,val}`.
  The existing raw fixed-frame cache is read but never deleted or overwritten.
- Initialize the radar encoder from the previous filtered-radar 25-epoch
  self-supervised pretrain, then train it jointly with a new generative decoder.
  The decoder uses the stride-8 latent without fine-resolution skip connections.
- Each generated 0.2 × 0.2 × 0.25 m fine voxel has four return slots. Each slot
  predicts presence, within-voxel XYZ offset, RCS, raw radial Doppler,
  compensated radial Doppler, and scan age. A frame with more than four returns
  in one voxel has a measurable reconstruction ceiling; the log reports the
  fraction of target points represented by candidate voxels and slots.

The loss is positive-weighted presence BCE over all generated slots, plus
Smooth L1 on positions and 0.1 × Smooth L1 on the remaining measured
attributes for present returns. Validation on all 1,296 frames reports exact
radar-voxel IoU, one-to-one radar-point F1 at 0.2 m, point precision/recall,
Chamfer squared distance, and the candidate/slot ceiling. The checkpoint is
selected by validation point F1 at the fixed 0.5 presence threshold.

The input itself is a perfect copy baseline (100% reconstruction by
definition). This trial measures what survives the bottleneck and learned
decoder. It does not test recovery of missing LiDAR or unseen radar returns.

## Professor machine

After fetching the publication commit into the copied radar checkout:

```bash
BASE=/mnt/3D10B36523559581/Gianluca
REPO="$BASE/sensor_fusion_outputs/Thesis-1-radar-autoencoder-prefilter-20261109"
REPO="$REPO" bash "$REPO/scripts/run_radar_self_autoencoder_professor.sh" all
```

The `all` phase builds only the new filtered radar cache, runs a 32/16-frame
GPU smoke test, trains 100 epochs, and sweeps validation thresholds on the
best checkpoint. `train` resumes from `model/last.ckpt`.
No old cache needs to be deleted. The official test cache from the previous
radar experiment is incomplete, so this launcher uses train and validation
only. Test should be scored after building its fixed radar frames.

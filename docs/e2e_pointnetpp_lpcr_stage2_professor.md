# PointNet++ radar + LiDAR LPCR + conditional Stage II

This is a new, isolated professor-machine run. It trains both encoders from
random initialization before Stage II. It does not reuse the laptop's radar
checkpoint or the older LPCR checkpoint.

1. Prepare aligned 20-sweep radar and clean LiDAR, corrected missing-only mild
   faults, the ≥12 distinct radar voxels within 1.6 m support masks, and the
   filtered Stage-II-cropped radar frames.
2. Train the radar-to-radar PointNet++ autoencoder for 300 epochs. It uses
   width 1024, 512 anchors, 2048 input/output returns, four children per
   anchor, seven radar columns, batch size 4, and no dropout. The best exact
   radar voxel IoU checkpoint is selected on validation.
3. Train the LiDAR encoder and LPCR decoder together for 30 epochs on clean
   LiDAR. The best full-region LPCR checkpoint is selected on validation.
4. Freeze the selected radar encoder. Cache its 512 anchor XYZ positions and
   1024-channel features for train, validation, and official test. The cache
   records the exact radar checkpoint hash. Empty radar frames yield empty
   feature sets. The train-only radar normalization is reused on every split.
5. Freeze LiDAR encoder and LPCR decoder. Train a new sparse diffusion U-Net
   and a 1024-to-latent radar adapter. The same fault-gated, unobserved
   ≥12/1.6 m active region is used. The loss is epsilon MSE plus standard
   generated occupancy BCE at weight 1.0. Full reconstruction is checked at
   fixed occupancy threshold 0.5 every five epochs, with one calibrated
   threshold sweep at the end. Validation and official test report full-scene
   and eligible missing-cell metrics, including exact and 0.2 m IoU.

At validation/test reconstruction, the model receives faulty LiDAR, measured
radar-derived frozen features, the radar gate, and Gaussian noise. Clean LiDAR
is loaded afterward for scoring. The current fault-area gate still uses the
synthetic injector footprint; this is an oracle dependency, so these results
do not establish real-world fault localization.

The launcher is `scripts/run_pointnetpp_lpcr_stage2_professor.sh`. Phases are
`prepare`, `radar`, `lidar`, `features`, `stage2`, and `all`. Each phase reuses
compatible cached data and resumes an existing checkpoint. Change `DATA`,
`RAW`, `REPO`, `UPSTREAM`, and other paths with environment variables if the
professor installation differs. The launcher needs the previously compiled
upstream D-PCC `pointops` module for PointNet++ farthest-point sampling; it
does not use the D-PCC autoencoder weights.

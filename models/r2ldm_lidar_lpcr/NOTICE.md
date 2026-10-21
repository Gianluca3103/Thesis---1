# R2LDM attribution

This independent LiDAR point-cloud reconstruction implementation adapts the
LPCR architecture and Stage-0 loss from
[NathanDrake67/R2LDM](https://github.com/NathanDrake67/R2LDM), commit
`4eac0a2c5a8a9c9aca44345a1d264fb184cb028f`. R2LDM is distributed under
Apache License 2.0; a copy is in [LICENSE-R2LDM](LICENSE-R2LDM).

The R2LDM sparse backbone builds on research software identified in its
[third-party notice](https://github.com/NathanDrake67/R2LDM/blob/main/THIRD_PARTY.md),
including Sparse2Dense and CenterPoint/Det3D. This module reimplements the
relevant architecture using `spconv.pytorch`; it does not copy their package
trees into this repository. Install spconv separately in the training environment.

The R2LDM release does not include pretrained weights or its exact paper
preprocessing. This implementation uses the official VoD train/validation
splits and raw LiDAR XYZ inside the released LPCR grid. It supports R2LDM's
corrected `centered_tanh` offset mode by default. The resulting checkpoints
are new and cannot be loaded into R2LDM's original classes without conversion.

"""Verify the separate SVEFusion detector environment before training."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sve-root", type=Path, required=True)
    args = parser.parse_args()
    sve_root = args.sve_root.resolve(strict=True)
    if not (sve_root / "pcdet" / "__init__.py").is_file():
        parser.error(f"Not an SVEFusion checkout: {sve_root}")
    sys.path.insert(0, str(sve_root))
    sys.path.insert(0, str(sve_root / "tools"))

    import easydict  # noqa: F401
    import numpy as np
    import spconv.pytorch  # noqa: F401
    import torch
    import yaml  # noqa: F401
    import causal_conv1d_cuda  # noqa: F401
    import selective_scan_cuda  # noqa: F401
    from mamba_ssm import Mamba
    from mamba_ssm.models.mixer_seq_simple import create_block  # noqa: F401
    from mamba_ssm.ops.triton.layernorm import RMSNorm  # noqa: F401
    import pcdet.datasets  # noqa: F401
    import pcdet.models  # noqa: F401
    from pcdet.ops.bev_pool import bev_pool_ext  # noqa: F401
    from pcdet.ops.ingroup_inds import ingroup_inds_cuda  # noqa: F401
    from pcdet.ops.iou3d_nms import iou3d_nms_cuda  # noqa: F401
    from pcdet.ops.pointnet2.pointnet2_batch import pointnet2_batch_cuda  # noqa: F401
    from pcdet.ops.pointnet2.pointnet2_stack import pointnet2_stack_cuda  # noqa: F401
    from pcdet.ops.roiaware_pool3d import roiaware_pool3d_cuda  # noqa: F401
    from pcdet.ops.roipoint_pool3d import roipoint_pool3d_cuda  # noqa: F401
    import svefusion_utils  # noqa: F401

    if sys.version_info[:2] != (3, 8):
        raise RuntimeError(f"SVEFusion requires Python 3.8, got {sys.version}")
    if torch.__version__.split("+")[0] != "2.1.0" or torch.version.cuda != "11.8":
        raise RuntimeError(
            f"Expected torch 2.1.0 with CUDA 11.8, got {torch.__version__} "
            f"with CUDA {torch.version.cuda}"
        )
    if np.__version__ != "1.23.5":
        raise RuntimeError(f"Expected NumPy 1.23.5, got {np.__version__}")
    if not torch.cuda.is_available():
        raise RuntimeError("PyTorch cannot use the NVIDIA GPU in this environment")
    result = torch.ones(1, device="cuda") * 2
    with torch.no_grad():
        sequence = torch.randn(1, 8, 16, device="cuda")
        reconstructed = Mamba(d_model=16, d_state=16, d_conv=4, expand=2).cuda()(sequence)
    torch.cuda.synchronize()
    if result.item() != 2 or not torch.isfinite(reconstructed).all().item():
        raise RuntimeError("CUDA or Mamba calculation failed")
    print(f"Detector Python: {sys.executable}")
    print(f"PyTorch: {torch.__version__}; CUDA: {torch.version.cuda}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print("SVEFusion Python dependencies, CUDA extensions, and GPU: OK")


if __name__ == "__main__":
    main()

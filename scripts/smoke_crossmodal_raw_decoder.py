"""GPU smoke test for the exact-PFE student's unpruned coordinate expansion."""

import json

import torch

from models.radar_lidar_crossmodal_v1.encoder import GRID
from models.radar_lidar_crossmodal_v1.student import RawSparseConfidenceDecoder, StudentConfig
from models.radar_lidar_stage1.sparse import SparseSites


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("The MinkowskiEngine decoder smoke test needs CUDA")
    device = "cuda"
    feature = torch.randn(1, 64, device=device, requires_grad=True)
    coarse = torch.tensor([[0, 0, 32, 32]], device=device)
    sites = SparseSites(coarse, feature, GRID.scale_shape(8), 8)
    decoder = RawSparseConfidenceDecoder(StudentConfig()).to(device)
    output = decoder(sites)
    expected_min = coarse[:, 1:] * 8
    expected_max = expected_min + 8
    if len(output.coords) != 512:
        raise AssertionError(f"Expected 512 generated children, got {len(output.coords)}")
    if not (((output.coords[:, 1:] >= expected_min) &
             (output.coords[:, 1:] < expected_max)).all()):
        raise AssertionError("Generated coordinates are outside their original bottleneck cell")
    if not torch.unique(output.coords, dim=0).shape[0] == 512:
        raise AssertionError("Generated child coordinates are not unique")
    output.features.square().mean().backward()
    if feature.grad is None or not torch.isfinite(feature.grad).all():
        raise AssertionError("Decoder gradient did not reach the radar bottleneck")
    print(json.dumps({"device": torch.cuda.get_device_name(0), "generated_sites": len(output.coords),
                      "stride": output.stride, "raw_logits": True, "pruning": False,
                      "input_gradient_finite": True}, indent=2))


if __name__ == "__main__":
    main()

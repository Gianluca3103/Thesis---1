"""GPU smoke test for raw stride-1 decoding and learned neighbor refinement."""

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
    pyramid = tuple(
        SparseSites(torch.tensor([[0, 0, 256 // stride, 256 // stride]], device=device),
                    torch.randn(1, channels, device=device), GRID.scale_shape(stride), stride)
        for stride, channels in ((1, 16), (2, 32), (4, 64)))
    output = decoder(sites, pyramid)
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
    if any(block.skip_projection.weight.grad is None or
           not torch.isfinite(block.skip_projection.weight.grad).all()
           for block in decoder.blocks):
        raise AssertionError("Fine radar levels did not reach the learned decoder")

    # A score just inside the second parent must depend on its neighbor.
    # With independent 8-cubed blocks this gradient is exactly zero.
    adjacent = torch.tensor([[0, 0, 32, 32], [0, 0, 32, 33]], device=device)
    adjacent_features = torch.randn(2, 64, device=device, requires_grad=True)
    neighboring = decoder(SparseSites(adjacent, adjacent_features, GRID.scale_shape(8), 8))
    edge = neighboring.coords[:, 3] == 33 * 8
    neighboring.features[edge].sum().backward()
    if not (adjacent_features.grad[0].abs().sum() > 1e-8):
        raise AssertionError("Raw scores cannot use features across coarse-cell boundaries")
    print(json.dumps({"device": torch.cuda.get_device_name(0), "generated_sites": len(output.coords),
                      "stride": output.stride, "raw_logits": True, "pruning": False,
                      "input_gradient_finite": True, "learned_radar_skips": True,
                      "cross_parent_refinement": True}, indent=2))


if __name__ == "__main__":
    main()

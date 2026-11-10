"""One synthetic GPU forward/backward through the radar point autoencoder."""

import torch

from models.radar_lidar_crossmodal_v1.radar_self_autoencoder import (
    RadarSelfAutoencoder, decoded_points, reconstruction_loss, target_slots)


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("Radar generative decoder smoke test requires CUDA")
    points = torch.zeros((1, 12, 7), device="cuda")
    for i in range(12):
        points[0, i, :3] = points.new_tensor((
            0.11 + 0.2 * (i % 4), 0.12 + 0.2 * (i // 4), 0.1))
        points[0, i, 3:] = points.new_tensor((1.0, 0.5, 0.4, 0.1))
    valid = torch.ones((1, 12), device="cuda", dtype=torch.bool)
    model = RadarSelfAutoencoder(None).cuda()
    prediction = model(points, valid)
    targets = target_slots(prediction, points, valid)
    loss, terms = reconstruction_loss(prediction, targets)
    if not loss.requires_grad or not torch.isfinite(loss):
        raise RuntimeError("Radar reconstruction loss is not finite and trainable")
    loss.backward()
    grads = [parameter.grad for parameter in model.parameters()
             if parameter.requires_grad and parameter.grad is not None]
    if not grads or not all(bool(torch.isfinite(grad).all()) for grad in grads):
        raise RuntimeError("Radar reconstruction gradients are absent or nonfinite")
    output, _ = decoded_points(prediction)
    print({"candidate_voxels": len(prediction.coords),
           "represented_points": targets["represented_points"],
           "target_points": targets["observed_points"],
           "predicted_points_at_0p5": len(output),
           "loss": float(loss.detach()),
           "bce": float(terms["bce"]),
           "offset": float(terms["offset"]),
           "attributes": float(terms["attributes"]),
           "finite_gradients": True})


if __name__ == "__main__":
    main()

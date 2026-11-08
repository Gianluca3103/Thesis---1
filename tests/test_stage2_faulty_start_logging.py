import torch

from scripts import train_stage2_conditional_faulty_start as trainer


def test_validation_loss_is_fixed_and_does_not_update_weights(monkeypatch):
    coarse = torch.tensor([[0, 0, 0, 0]])
    hierarchy = (coarse,) * 5
    monkeypatch.setattr(trainer, "_batch", lambda rows, device: (
        hierarchy, torch.ones((1, 3)), torch.zeros(1, dtype=torch.long)))
    monkeypatch.setattr(trainer, "_conditioned_inputs", lambda *args, **kwargs: (
        torch.zeros((1, 32)), torch.zeros(1, dtype=torch.bool),
        torch.zeros((1, 32))))

    class Encoder:
        def __call__(self, clean, coarse, ids):
            return torch.ones((1, 32)), torch.ones(1, dtype=torch.bool)

    class Diffusion(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(0.1))

        def forward(self, *args):
            return self.weight.expand(1, 32)

    class Model:
        lidar_encoder = Encoder()
        diffusion = Diffusion()
        radar_adapter = torch.nn.Linear(1, 1)

        @staticmethod
        def eval():
            return None

    model = Model()
    alpha = torch.full((1000,), 0.5)
    loader = [[{"frame_id": "00164"}]]
    initial = model.diffusion.weight.detach().clone()
    first = trainer.loss_epoch(model, loader, None, alpha, device="cpu",
                               epoch=1, seed=3103, timestep=499)
    second = trainer.loss_epoch(model, loader, None, alpha, device="cpu",
                                epoch=99, seed=3103, timestep=499)
    assert first == second
    assert torch.equal(model.diffusion.weight, initial)
    assert model.diffusion.weight.grad is None

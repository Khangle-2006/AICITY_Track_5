from __future__ import annotations


def import_torch():
    try:
        import torch  # type: ignore
        import torch.nn as nn  # type: ignore
        import torch.nn.functional as F  # type: ignore

        return torch, nn, F
    except Exception as exc:
        raise SystemExit(f"PyTorch is required for the trainable refiner. Original error: {exc}")


class TinyResidualRefiner:
    """Factory wrapper to avoid importing torch at module import time."""

    @staticmethod
    def build(in_channels: int = 13, width: int = 48, residual_scale: float = 0.25):
        torch, nn, _ = import_torch()

        class _Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.residual_scale = residual_scale
                self.net = nn.Sequential(
                    nn.Conv2d(in_channels, width, 3, padding=1),
                    nn.SiLU(inplace=True),
                    nn.Conv2d(width, width, 3, padding=1),
                    nn.SiLU(inplace=True),
                    nn.Conv2d(width, width, 3, padding=1),
                    nn.SiLU(inplace=True),
                    nn.Conv2d(width, 3, 3, padding=1),
                    nn.Tanh(),
                )

            def forward(self, x, base_rgb):
                residual = self.net(x) * self.residual_scale
                return (base_rgb + residual).clamp(0.0, 1.0)

        return _Model()


def gradient_loss(torch, pred, target):
    pred_dx = pred[:, :, :, 1:] - pred[:, :, :, :-1]
    target_dx = target[:, :, :, 1:] - target[:, :, :, :-1]
    pred_dy = pred[:, :, 1:, :] - pred[:, :, :-1, :]
    target_dy = target[:, :, 1:, :] - target[:, :, :-1, :]
    return (pred_dx - target_dx).abs().mean() + (pred_dy - target_dy).abs().mean()


def refiner_loss(torch, pred, target, base, lambda_grad: float = 0.10, lambda_residual: float = 0.02):
    l1 = (pred - target).abs().mean()
    grad = gradient_loss(torch, pred, target)
    residual = (pred - base).abs().mean()
    return l1 + lambda_grad * grad + lambda_residual * residual, {
        "l1": float(l1.detach().cpu()),
        "grad": float(grad.detach().cpu()),
        "residual": float(residual.detach().cpu()),
    }

"""Complex-compressed spectral MSE loss (ccMSE): a weighted sum of a
magnitude-MSE term and a complex (real+imaginary)-MSE term, with an optional
power-law compression on the magnitude (``gamma``)."""

from typing import Final

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.autograd import Function


def as_complex(x: Tensor) -> Tensor:
    """Accepts either a native complex tensor or a real tensor whose last
    dimension is 2 (real, imag) and returns a complex tensor."""
    if torch.is_complex(x):
        return x
    if x.shape[-1] != 2:
        raise ValueError(f"Last dimension must be of length 2 (re + im), got {x.shape}")
    if x.stride(-1) != 1:
        x = x.contiguous()
    return torch.view_as_complex(x)


class _Angle(Function):
    """torch.angle with a gradient that is robustified for near-zero
    magnitude (avoids exploding gradients at the origin of the complex
    plane)."""

    @staticmethod
    def forward(ctx, x: Tensor) -> Tensor:
        ctx.save_for_backward(x)
        return torch.atan2(x.imag, x.real)

    @staticmethod
    def backward(ctx, grad: Tensor) -> Tensor:
        (x,) = ctx.saved_tensors
        grad_inv = grad / (x.real.square() + x.imag.square()).clamp_min_(1e-10)
        return torch.view_as_complex(torch.stack((-x.imag * grad_inv, x.real * grad_inv), dim=-1))


class SpectralLoss(nn.Module):
    """ccMSE loss between a complex STFT estimate and target.

    loss = factor_magnitude * MSE(|est|^gamma, |target|^gamma)
         + factor_complex   * MSE(est_compressed, target_compressed)
    """

    gamma: Final[float]
    factor_magnitude: Final[float]
    factor_complex: Final[float]

    def __init__(self, gamma: float = 1.0, factor_magnitude: float = 1.0, factor_complex: float = 1.0):
        super().__init__()
        self.gamma = gamma
        self.factor_magnitude = factor_magnitude
        self.factor_complex = factor_complex

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        input = as_complex(input)
        target = as_complex(target)

        input_mag = input.abs()
        target_mag = target.abs()
        if self.gamma != 1:
            input_mag = input_mag.clamp_min(1e-12).pow(self.gamma)
            target_mag = target_mag.clamp_min(1e-12).pow(self.gamma)

        loss = F.mse_loss(input_mag, target_mag) * self.factor_magnitude

        if self.factor_complex > 0:
            if self.gamma != 1:
                input = input_mag * torch.exp(1j * _Angle.apply(input))
                target = target_mag * torch.exp(1j * _Angle.apply(target))
            loss_complex = F.mse_loss(torch.view_as_real(input), torch.view_as_real(target)) * self.factor_complex
            loss = loss + loss_complex

        return loss

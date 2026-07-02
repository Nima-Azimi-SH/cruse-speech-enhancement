"""STFT / iSTFT utilities and feature extraction shared by training and
evaluation."""

import torch


def make_sqrt_hann_window(win_length: int, device=None) -> torch.Tensor:
    """Square-root-Hann analysis/synthesis window (needed for perfect
    reconstruction with 50% overlap when the same window is used for both
    STFT and iSTFT)."""
    w = torch.hann_window(win_length, periodic=True, device=device)
    return torch.sqrt(w)


def stft_batch(
    x: torch.Tensor,
    n_fft: int,
    hop_length: int,
    win_length: int,
    window: torch.Tensor,
) -> torch.Tensor:
    """Batched STFT. x: (B, L) -> X: (B, F, T) complex."""
    return torch.stft(
        x,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
        center=True,
        return_complex=True,
    )


def istft_batch(
    X: torch.Tensor,
    n_fft: int,
    hop_length: int,
    win_length: int,
    window: torch.Tensor,
    length: int,
) -> torch.Tensor:
    """Batched inverse STFT. X: (B, F, T) complex -> x: (B, L)."""
    return torch.istft(
        X,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
        center=True,
        length=length,
    )


def log_power_spec(X: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """log(|X|^2 + eps), used as the network's input feature. X: (B, F, T)
    complex -> (B, F, T) real."""
    power = X.real**2 + X.imag**2
    return torch.log(power + eps)

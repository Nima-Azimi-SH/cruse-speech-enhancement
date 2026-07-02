"""Evaluation metrics."""

import torch


def si_sdr(est: torch.Tensor, ref: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Scale-Invariant Signal-to-Distortion Ratio (SI-SDR), computed per
    sample, in dB.

    est, ref: (B, L) tensors -> returns: (B,)
    """
    est_zm = est - est.mean(dim=-1, keepdim=True)
    ref_zm = ref - ref.mean(dim=-1, keepdim=True)

    ref_energy = (ref_zm**2).sum(dim=-1, keepdim=True) + eps
    alpha = (est_zm * ref_zm).sum(dim=-1, keepdim=True) / ref_energy
    proj = alpha * ref_zm
    noise = est_zm - proj

    ratio = (proj**2).sum(dim=-1) / ((noise**2).sum(dim=-1) + eps)
    return 10.0 * torch.log10(ratio + eps)

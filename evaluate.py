#!/usr/bin/env python3
"""Evaluate a trained CRUSE checkpoint on a test set (SI-SDR of noisy vs.
enhanced mixtures), and optionally export one example's noisy/enhanced/clean
audio for a quick listen.

Example
-------
python evaluate.py \\
    --clean_test_folder Signals/Clean_Test \\
    --noise_folder Signals/Noise \\
    --checkpoint checkpoints/cruse_best.pth \\
    --export_example_dir assets/example_audio
"""

import argparse
import os

import numpy as np
import torch
from scipy.io import wavfile

from src.audio import istft_batch, log_power_spec, make_sqrt_hann_window, stft_batch
from src.data import dataloader_test
from src.metrics import si_sdr
from src.model import CRUSE


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--clean_test_folder", required=True)
    p.add_argument("--noise_folder", required=True)
    p.add_argument("--checkpoint", default="checkpoints/cruse_best.pth")
    p.add_argument("--fs", type=int, default=16000)
    p.add_argument("--n_fft", type=int, default=256)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--export_example_dir", default=None, help="If set, saves one noisy/enhanced/clean wav triplet here")
    p.add_argument("--device", default=None)
    return p.parse_args()


def enhance(model, noisy, n_fft, hop_length, win_length, window):
    length = noisy.shape[-1]
    xn = stft_batch(noisy, n_fft, hop_length, win_length, window)
    feature = log_power_spec(xn).permute(0, 2, 1).unsqueeze(1)
    gain = model(feature).squeeze(1).permute(0, 2, 1).contiguous()
    xhat = xn * gain
    return istft_batch(xhat, n_fft, hop_length, win_length, window, length=length)


def main():
    args = parse_args()
    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    win_length = int(0.016 * args.fs)
    hop_length = win_length // 2
    window = make_sqrt_hann_window(win_length, device=device)

    model = CRUSE(input_freq_bins=args.n_fft // 2 + 1).to(device)
    state = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state)
    model.eval()

    test_loader = dataloader_test(args.clean_test_folder, args.noise_folder, args.fs, args.batch_size, shuffle=False)

    noisy_scores, enh_scores = [], []
    example_saved = False

    with torch.no_grad():
        for noisy, clean in test_loader:
            noisy, clean = noisy.to(device), clean.to(device)
            xhat = enhance(model, noisy, args.n_fft, hop_length, win_length, window)

            noisy_scores.append(si_sdr(noisy, clean).cpu())
            enh_scores.append(si_sdr(xhat, clean).cpu())

            if args.export_example_dir and not example_saved:
                os.makedirs(args.export_example_dir, exist_ok=True)
                wavfile.write(os.path.join(args.export_example_dir, "noisy.wav"), args.fs, noisy[0].cpu().numpy().astype(np.float32))
                wavfile.write(os.path.join(args.export_example_dir, "enhanced.wav"), args.fs, xhat[0].cpu().numpy().astype(np.float32))
                wavfile.write(os.path.join(args.export_example_dir, "clean.wav"), args.fs, clean[0].cpu().numpy().astype(np.float32))
                example_saved = True

    noisy_scores = torch.cat(noisy_scores, dim=0)
    enh_scores = torch.cat(enh_scores, dim=0)
    improvement = enh_scores - noisy_scores

    results = {
        "noisy_si_sdr_mean": noisy_scores.mean().item(),
        "noisy_si_sdr_var": noisy_scores.var(unbiased=False).item(),
        "enh_si_sdr_mean": enh_scores.mean().item(),
        "enh_si_sdr_var": enh_scores.var(unbiased=False).item(),
        "improvement_mean": improvement.mean().item(),
        "improvement_var": improvement.var(unbiased=False).item(),
        "num_examples": int(noisy_scores.numel()),
    }

    print("----- TEST RESULTS -----")
    for k, v in results.items():
        print(f"{k}: {v}")

    if args.export_example_dir:
        print(f"\nSaved one example triplet (noisy/enhanced/clean) to {args.export_example_dir}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Train CRUSE for speech enhancement.

Example
-------
python train.py \\
    --clean_train_folder Signals/Clean_Train \\
    --clean_val_folder Signals/Clean_Val \\
    --noise_folder Signals/Noise \\
    --save_path checkpoints/cruse_best.pth \\
    --num_epochs 100
"""

import argparse

import torch

from src.audio import istft_batch, log_power_spec, make_sqrt_hann_window, stft_batch
from src.data import dataloader_train, dataloader_val
from src.losses import SpectralLoss
from src.model import CRUSE


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--clean_train_folder", required=True, help="Folder of clean training .wav files")
    p.add_argument("--clean_val_folder", required=True, help="Folder of clean validation .wav files")
    p.add_argument("--noise_folder", required=True, help="Folder of noise .wav files (shared by train/val)")
    p.add_argument("--save_path", default="checkpoints/cruse_best.pth", help="Where to save the best checkpoint")
    p.add_argument("--fs", type=int, default=16000, help="Sampling rate (Hz)")
    p.add_argument("--n_fft", type=int, default=256, help="STFT FFT size")
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--num_epochs", type=int, default=100)
    p.add_argument("--lr", type=float, default=8e-5)
    p.add_argument("--weight_decay", type=float, default=0.1)
    p.add_argument("--log_every", type=int, default=5, help="Print train/val loss every N epochs")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default=None, help="cuda / cpu (default: auto-detect)")
    return p.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    win_length = int(0.016 * args.fs)  # 16 ms
    hop_length = win_length // 2  # 50% overlap
    window = make_sqrt_hann_window(win_length, device=device)

    train_loader = dataloader_train(args.clean_train_folder, args.noise_folder, args.fs, args.batch_size, shuffle=True)
    val_loader = dataloader_val(args.clean_val_folder, args.noise_folder, args.fs, args.batch_size, shuffle=False)

    model = CRUSE(input_freq_bins=args.n_fft // 2 + 1).to(device)
    criterion = SpectralLoss()
    # model.parameters() already includes every submodule (including the
    # bottleneck GRU) at this point, since CRUSE builds them in __init__.
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_val = float("inf")

    for epoch in range(args.num_epochs):
        model.train()
        train_loss_sum, train_batches = 0.0, 0
        for noisy, clean in train_loader:
            noisy, clean = noisy.to(device), clean.to(device)
            length = noisy.shape[-1]

            xn = stft_batch(noisy, args.n_fft, hop_length, win_length, window)
            xc = stft_batch(clean, args.n_fft, hop_length, win_length, window)

            feature = log_power_spec(xn).permute(0, 2, 1).unsqueeze(1)  # (B, 1, T, F)
            gain = model(feature).squeeze(1).permute(0, 2, 1).contiguous()  # (B, F, T)

            xhat = xn * gain
            xhat_time = istft_batch(xhat, args.n_fft, hop_length, win_length, window, length=length)
            xhat_consistent = stft_batch(xhat_time, args.n_fft, hop_length, win_length, window)

            loss = criterion(xhat_consistent, xc)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss_sum += loss.item()
            train_batches += 1

        avg_train = train_loss_sum / max(train_batches, 1)

        model.eval()
        val_loss_sum, val_batches = 0.0, 0
        with torch.no_grad():
            for noisy, clean in val_loader:
                noisy, clean = noisy.to(device), clean.to(device)
                length = noisy.shape[-1]

                xn = stft_batch(noisy, args.n_fft, hop_length, win_length, window)
                xc = stft_batch(clean, args.n_fft, hop_length, win_length, window)

                feature = log_power_spec(xn).permute(0, 2, 1).unsqueeze(1)
                gain = model(feature).squeeze(1).permute(0, 2, 1).contiguous()

                xhat = xn * gain
                xhat_time = istft_batch(xhat, args.n_fft, hop_length, win_length, window, length=length)
                xhat_consistent = stft_batch(xhat_time, args.n_fft, hop_length, win_length, window)

                val_loss_sum += criterion(xhat_consistent, xc).item()
                val_batches += 1

        avg_val = val_loss_sum / max(val_batches, 1)

        if (epoch + 1) % args.log_every == 0 or epoch == 0:
            print(f"Epoch {epoch + 1}/{args.num_epochs} | Train loss: {avg_train:.6f} | Val loss: {avg_val:.6f}")

        if avg_val < best_val:
            best_val = avg_val
            torch.save(model.state_dict(), args.save_path)

    print("Training finished.")
    print(f"Best validation loss: {best_val:.6f}")


if __name__ == "__main__":
    main()

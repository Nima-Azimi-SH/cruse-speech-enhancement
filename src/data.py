"""On-the-fly noisy speech mixture generation and dataloaders.

Clean speech and noise recordings are mixed at a random SNR, following the
same data-generation logic as the reference exercise: for training, an SNR
is drawn from N(mean=5 dB, var=10); for validation/test, a fixed 5 dB SNR
and a deterministic (first-N-seconds) crop are used for reproducibility.

NOTE: this repository does not ship any audio data. Point ``clean_folder``
and ``noise_folder`` at your own directories of 16 kHz mono (or resampled)
``.wav`` files -- see the README for suggested public sources (e.g. the
Microsoft DNS-Challenge speech/noise corpora, which the CRUSE paper itself
trains on).
"""

import glob
import os

import numpy as np
import torch
from scipy.io import wavfile
from torch.utils.data import DataLoader, Dataset


def load_and_preprocess(file_path: str, num_samples: int, random_crop: bool = True):
    """Read a wav file, downmix to mono, and crop/pad to a fixed length,
    normalized to [-1, 1].

    random_crop:
        True  -> if longer than num_samples, cut a RANDOM segment (training)
        False -> if longer than num_samples, cut the FIRST segment (val/test)
    """
    fs, x = wavfile.read(file_path)

    if x.ndim == 2:  # stereo -> mono
        x = x.mean(axis=1)
    x = x.astype(np.float32)

    if len(x) > num_samples:
        if random_crop:
            start = np.random.randint(0, len(x) - num_samples + 1)
            x = x[start : start + num_samples]
        else:
            x = x[:num_samples]
    elif len(x) < num_samples:
        x = np.pad(x, (0, num_samples - len(x)))

    max_abs = np.max(np.abs(x)) + 1e-8
    x = x / max_abs
    return fs, x


def scale_noise_to_snr(clean: np.ndarray, noise: np.ndarray, snr_db: float) -> np.ndarray:
    """Scale ``noise`` so that mixing it with ``clean`` yields the target SNR
    (in dB), scaling the noise only.

        SNR(dB) = 10 * log10(P_speech / P_noise_scaled)
        alpha   = sqrt(P_noise_scaled / P_noise)   (power ~ amplitude^2)
    """
    p_speech = np.mean(clean**2) + 1e-8
    p_noise = np.mean(noise**2) + 1e-8
    snr_linear = 10 ** (snr_db / 10)
    p_noise_target = p_speech / snr_linear
    alpha = np.sqrt(p_noise_target / p_noise)
    return alpha * noise


class SpeechNoiseMixDataset(Dataset):
    """Pairs every clean utterance with every noise file (N_clean x N_noise
    combinations per epoch) and mixes them on-the-fly at a random or fixed
    SNR."""

    def __init__(self, clean_folder: str, noise_folder: str, fs: int, duration_sec: float = 10, is_validation: bool = False):
        self.fs = fs
        self.num_samples = int(fs * duration_sec)
        self.is_validation = is_validation

        self.clean_list = sorted(glob.glob(os.path.join(clean_folder, "*.wav")))
        self.noise_list = sorted(glob.glob(os.path.join(noise_folder, "*.wav")))

        if not self.clean_list:
            raise ValueError(f"No clean .wav files found in '{clean_folder}'.")
        if not self.noise_list:
            raise ValueError(f"No noise .wav files found in '{noise_folder}'.")

        self.n_clean = len(self.clean_list)
        self.n_noise = len(self.noise_list)

    def __len__(self):
        return self.n_clean * self.n_noise

    def __getitem__(self, idx):
        clean_idx = idx // self.n_noise
        noise_idx = idx % self.n_noise

        random_crop = not self.is_validation
        _, clean = load_and_preprocess(self.clean_list[clean_idx], self.num_samples, random_crop=random_crop)
        _, noise = load_and_preprocess(self.noise_list[noise_idx], self.num_samples, random_crop=random_crop)

        snr_db = 5.0 if self.is_validation else np.random.normal(loc=5.0, scale=np.sqrt(10.0))

        noise_scaled = scale_noise_to_snr(clean, noise, snr_db)
        noisy = clean + noise_scaled

        return torch.from_numpy(noisy).float(), torch.from_numpy(clean).float()


def make_dataloader(clean_folder: str, noise_folder: str, fs: int, batch_size: int, is_validation: bool, shuffle: bool) -> DataLoader:
    dataset = SpeechNoiseMixDataset(clean_folder, noise_folder, fs, duration_sec=10, is_validation=is_validation)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def dataloader_train(clean_folder: str, noise_folder: str, fs: int, batch_size: int, shuffle: bool = True) -> DataLoader:
    return make_dataloader(clean_folder, noise_folder, fs, batch_size, is_validation=False, shuffle=shuffle)


def dataloader_val(clean_folder: str, noise_folder: str, fs: int, batch_size: int, shuffle: bool = False) -> DataLoader:
    return make_dataloader(clean_folder, noise_folder, fs, batch_size, is_validation=True, shuffle=shuffle)


def dataloader_test(clean_folder: str, noise_folder: str, fs: int, batch_size: int, shuffle: bool = False) -> DataLoader:
    return make_dataloader(clean_folder, noise_folder, fs, batch_size, is_validation=True, shuffle=shuffle)

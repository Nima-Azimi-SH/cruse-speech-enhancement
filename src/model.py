"""
CRUSE model architecture.

Reproduction of the CRUSE (Convolutional Recurrent U-net for Speech Enhancement)
network from:

    S. Braun, H. Gamper, C. K. A. Reddy, I. Tashev,
    "Towards Efficient Models for Real-Time Deep Noise Suppression,"
    ICASSP 2021.

This implementation follows the CRUSE4-128-1xGRU4 configuration: 4 convolutional
encoder/decoder layers (filters 16-32-64-128), 1x1 conv skip connections, and a
bottleneck of 4 parallel single-layer GRUs.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def pad_time_freq(x: torch.Tensor) -> torch.Tensor:
    """Asymmetric padding so that stride-(1, 2) conv keeps the time axis
    unchanged and halves the frequency axis.

    Input tensor layout is (B, C, T, F). ``F.pad`` pads the last two dims in
    (left, right, top, bottom) order, i.e. (F_left, F_right, T_top, T_bottom).
    """
    return F.pad(x, (1, 1, 0, 1))


class EncoderConvBlock(nn.Module):
    """Conv2d (kernel=(2,3), stride=(1,2)) + LeakyReLU."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels=in_ch,
            out_channels=out_ch,
            kernel_size=(2, 3),
            stride=(1, 2),
            padding=0,  # explicit asymmetric padding is applied in forward()
        )
        self.act = nn.LeakyReLU(negative_slope=0.2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = pad_time_freq(x)
        x = self.conv(x)
        return self.act(x)


class DecoderDeconvBlock(nn.Module):
    """ConvTranspose2d (kernel=(2,3), stride=(1,2)) + LeakyReLU/Sigmoid.

    Transpose convolutions on odd-sized inputs (e.g. 129 freq bins) can
    produce an output that is one bin larger than the matching encoder skip
    connection. The output is cropped to the target shape when needed.
    """

    def __init__(self, in_ch: int, out_ch: int, is_last: bool = False):
        super().__init__()
        self.deconv = nn.ConvTranspose2d(
            in_channels=in_ch,
            out_channels=out_ch,
            kernel_size=(2, 3),
            stride=(1, 2),
            padding=0,
        )
        self.act = nn.Sigmoid() if is_last else nn.LeakyReLU(negative_slope=0.2)

    def forward(self, x: torch.Tensor, target_t: int = None, target_f: int = None) -> torch.Tensor:
        x = self.deconv(x)
        x = self.act(x)
        if target_t is not None and x.shape[-2] != target_t:
            x = x[:, :, :target_t, :]
        if target_f is not None and x.shape[-1] != target_f:
            x = x[:, :, :, :target_f]
        return x


class ParallelGRUBlock(nn.Module):
    """Bottleneck recurrent block: splits the flattened (channel x freq)
    feature dimension into ``num_parallel`` equal parts and runs one
    single-layer GRU per part, following the "parallel GRU" design used to
    keep CRUSE's bottleneck cheap.

    Input/output shape: (B, C, T, F).
    """

    def __init__(self, channels: int, freq_bins: int, num_parallel: int = 4):
        super().__init__()
        self.channels = channels
        self.freq_bins = freq_bins
        self.num_parallel = num_parallel

        total_feat = channels * freq_bins
        assert total_feat % num_parallel == 0, (
            f"channels*freq_bins ({total_feat}) must be divisible by "
            f"num_parallel ({num_parallel})"
        )
        self.part_feat = total_feat // num_parallel

        self.grus = nn.ModuleList(
            [
                nn.GRU(
                    input_size=self.part_feat,
                    hidden_size=self.part_feat,
                    num_layers=1,
                    batch_first=True,
                )
                for _ in range(num_parallel)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, t, f = x.shape
        x = x.permute(0, 2, 1, 3).contiguous().view(b, t, c * f)  # (B, T, C*F)

        parts = torch.split(x, self.part_feat, dim=-1)
        out_parts = [gru(part)[0] for gru, part in zip(self.grus, parts)]

        x_out = torch.cat(out_parts, dim=-1)  # (B, T, C*F)
        x_out = x_out.view(b, t, c, f).permute(0, 2, 1, 3).contiguous()  # (B, C, T, F)
        return x_out


def _encoder_output_freq_bins(freq_bins_in: int, num_layers: int) -> int:
    """Analytically compute the frequency-bin count after ``num_layers``
    stride-(1,2) encoder blocks (see ``pad_time_freq`` / ``EncoderConvBlock``).

    For each layer: F_padded = F + 2, F_out = floor((F_padded - 3) / 2) + 1
                                            = (F - 1) // 2 + 1
    """
    f = freq_bins_in
    for _ in range(num_layers):
        f = (f - 1) // 2 + 1
    return f


class CRUSE(nn.Module):
    """CRUSE4-128-1xGRU4: 4 encoder/decoder layers (16-32-64-128 filters),
    1x1 conv skip connections, and a bottleneck of 4 parallel GRUs.

    Parameters
    ----------
    input_freq_bins:
        Number of STFT frequency bins fed to the network (e.g. 129 for a
        256-point FFT). Used to size the bottleneck GRUs analytically, so
        that all submodules exist at construction time -- this matters
        because it lets an optimizer created right after ``CRUSE()`` see
        every trainable parameter, including the GRUs, from the start.
    num_parallel_gru:
        Number of parallel GRUs in the bottleneck (P in the paper's
        CRUSE-L-C_L-NxGRU-P naming).
    """

    def __init__(self, input_freq_bins: int = 129, num_parallel_gru: int = 4):
        super().__init__()
        num_enc_layers = 4

        # Encoder channels: 1 -> 16 -> 32 -> 64 -> 128
        self.enc1 = EncoderConvBlock(1, 16)
        self.enc2 = EncoderConvBlock(16, 32)
        self.enc3 = EncoderConvBlock(32, 64)
        self.enc4 = EncoderConvBlock(64, 128)

        # 1x1 conv skip connections
        self.skip1 = nn.Conv2d(16, 16, kernel_size=1)
        self.skip2 = nn.Conv2d(32, 32, kernel_size=1)
        self.skip3 = nn.Conv2d(64, 64, kernel_size=1)
        self.skip4 = nn.Conv2d(128, 128, kernel_size=1)

        # Decoder channels: 128 -> 64 -> 32 -> 16 -> 1
        self.dec1 = DecoderDeconvBlock(128, 64)
        self.dec2 = DecoderDeconvBlock(64, 32)
        self.dec3 = DecoderDeconvBlock(32, 16)
        self.dec4 = DecoderDeconvBlock(16, 1, is_last=True)  # sigmoid -> gain in [0, 1]

        # Bottleneck GRUs, sized analytically so they are registered
        # (and visible to any optimizer built right after construction)
        # before the first forward pass.
        bottleneck_freq_bins = _encoder_output_freq_bins(input_freq_bins, num_enc_layers)
        self.gru = ParallelGRUBlock(channels=128, freq_bins=bottleneck_freq_bins, num_parallel=num_parallel_gru)

        self.input_freq_bins = input_freq_bins

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 1, T, F)
        _, _, t, f = x.shape

        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)

        s1 = self.skip1(e1)
        s2 = self.skip2(e2)
        s3 = self.skip3(e3)
        s4 = self.skip4(e4)

        b = self.gru(e4)

        d1 = self.dec1(b + s4, target_t=s3.shape[-2], target_f=s3.shape[-1])
        d2 = self.dec2(d1 + s3, target_t=s2.shape[-2], target_f=s2.shape[-1])
        d3 = self.dec3(d2 + s2, target_t=s1.shape[-2], target_f=s1.shape[-1])
        d4 = self.dec4(d3 + s1, target_t=t, target_f=f)
        return d4

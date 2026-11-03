"""Audio loading and log-Mel filterbank front-end.

The CaptionStew CLAP models were trained on 80-dim Kaldi-compatible log-Mel
filterbanks computed on the fly by Lhotse (``Fbank(FbankConfig(num_mel_bins=80))``:
16 kHz, 25 ms window, 10 ms shift, povey window, pre-emphasis 0.97, DC offset
removal, ``snip_edges=False``, no dither). This module re-implements that exact
front-end in plain PyTorch (ported from ``lhotse.features.kaldi.layers``,
Apache-2.0) so that inference does not depend on Lhotse.
"""

from __future__ import annotations

import math
from typing import List, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence

LOG_EPSILON = math.log(1e-10)  # padding value used by Lhotse when collating features

AudioInput = Union[str, np.ndarray, torch.Tensor]


def _lin2mel(x: torch.Tensor) -> torch.Tensor:
    return 1127.0 * torch.log(1 + x / 700.0)


def _mel_banks(
    num_bins: int, fft_length: int, sample_freq: float, low_freq: float, high_freq: float
) -> torch.Tensor:
    """Kaldi/torchaudio-compatible triangular mel filters, shape (num_fft_bins + 1, num_bins)."""
    num_fft_bins = fft_length // 2
    nyquist = 0.5 * sample_freq
    if high_freq <= 0.0:
        high_freq += nyquist
    fft_bin_width = sample_freq / fft_length
    mel_low = _lin2mel(torch.tensor(low_freq, dtype=torch.float64))
    mel_high = _lin2mel(torch.tensor(high_freq, dtype=torch.float64))
    delta = (mel_high - mel_low) / (num_bins + 1)
    b = torch.arange(num_bins, dtype=torch.float64).unsqueeze(1)
    left = mel_low + b * delta
    center = mel_low + (b + 1.0) * delta
    right = mel_low + (b + 2.0) * delta
    mel = _lin2mel(fft_bin_width * torch.arange(num_fft_bins, dtype=torch.float64)).unsqueeze(0)
    up = (mel - left) / (center - left)
    down = (right - mel) / (right - center)
    bins = torch.clamp(torch.min(up, down), min=0.0).to(torch.float32)
    # pad the Nyquist bin with zeros, transpose to (num_fft_bins + 1, num_bins)
    return torch.nn.functional.pad(bins, (0, 1)).T.contiguous()


class Fbank(nn.Module):
    """Batched Kaldi-style log-Mel filterbank, numerically matching Lhotse's ``Fbank``.

    Input: waveform tensor of shape ``(num_samples,)`` or ``(batch, num_samples)``
    (all items in a batch must have the same length; use :func:`compute_fbank` for
    variable-length inputs). Output: ``(batch, num_frames, num_mel_bins)``.
    """

    def __init__(
        self,
        sampling_rate: int = 16000,
        num_mel_bins: int = 80,
        frame_length: float = 0.025,
        frame_shift: float = 0.01,
        preemph_coeff: float = 0.97,
        low_freq: float = 20.0,
        high_freq: float = -400.0,
    ):
        super().__init__()
        self.sampling_rate = sampling_rate
        self.win_length = int(math.floor(frame_length * sampling_rate))
        self.hop_length = int(math.floor(frame_shift * sampling_rate))
        self.fft_length = 1 << (self.win_length - 1).bit_length()
        self.preemph_coeff = preemph_coeff
        window = torch.hann_window(self.win_length, periodic=False).pow(0.85)  # povey
        self.register_buffer("window", window, persistent=False)
        self.register_buffer(
            "mel_banks",
            _mel_banks(num_mel_bins, self.fft_length, sampling_rate, low_freq, high_freq),
            persistent=False,
        )
        self._eps = torch.finfo(torch.float32).eps

    def num_frames(self, num_samples: int) -> int:
        """Number of frames for ``snip_edges=False`` (depends only on the hop size)."""
        return (num_samples + self.hop_length // 2) // self.hop_length

    def _frames(self, x: torch.Tensor) -> torch.Tensor:
        num_samples = x.size(-1)
        n = self.num_frames(num_samples)
        npad = (n - 1) * self.hop_length + self.win_length - num_samples
        npad_left = (self.win_length - self.hop_length) // 2
        npad_right = npad - npad_left
        parts = [torch.flip(x[:, :npad_left], (1,)), x]
        if npad_right > 0:
            parts.append(torch.flip(x[:, -npad_right:], (1,)))
        x = torch.cat(parts, dim=1)
        if npad_right < 0:
            x = x[:, : x.size(1) + npad_right]
        return x.unfold(1, self.win_length, self.hop_length)  # (B, n, win_length)

    @torch.no_grad()
    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        if wav.dim() == 1:
            wav = wav.unsqueeze(0)
        x = self._frames(wav.to(torch.float32))
        x = x - x.mean(dim=-1, keepdim=True)  # remove DC offset
        prev = torch.nn.functional.pad(x, (1, 0), mode="replicate")[..., :-1]
        # buffers may have been cast by model.half(); the front-end always runs in fp32
        x = (x - self.preemph_coeff * prev) * self.window.float()
        x = torch.nn.functional.pad(x, (0, self.fft_length - self.win_length))
        power = torch.fft.rfft(x, dim=-1).abs().pow(2)
        mel = torch.matmul(power, self.mel_banks.float())
        return torch.clamp(mel, min=self._eps).log()


def load_audio(path: str, target_sr: int = 16000) -> torch.Tensor:
    """Read an audio file as a mono float32 waveform at ``target_sr``.

    Multi-channel audio is down-mixed by averaging the channels.
    """
    import soundfile as sf

    try:
        data, sr = sf.read(path, dtype="float32", always_2d=True)
        wav = torch.from_numpy(data.mean(axis=1))
    except Exception:
        # Formats libsndfile cannot decode (e.g. some mp3/m4a): fall back to torchaudio.
        import torchaudio

        wav, sr = torchaudio.load(path)
        wav = wav.mean(dim=0)
    return resample(wav, sr, target_sr)


def resample(wav: torch.Tensor, orig_sr: int, target_sr: int) -> torch.Tensor:
    if orig_sr == target_sr:
        return wav
    try:
        import torchaudio.functional as AF

        return AF.resample(wav, orig_freq=orig_sr, new_freq=target_sr)
    except Exception:
        from scipy.signal import resample_poly

        g = math.gcd(int(orig_sr), int(target_sr))
        out = resample_poly(wav.numpy(), target_sr // g, orig_sr // g)
        return torch.from_numpy(np.ascontiguousarray(out, dtype=np.float32))


def to_waveform(item: AudioInput, sampling_rate: int = 16000) -> torch.Tensor:
    """Turn a path / 1-D numpy array / 1-D tensor into a 1-D float32 tensor."""
    if isinstance(item, str):
        return load_audio(item, sampling_rate)
    if isinstance(item, np.ndarray):
        item = torch.from_numpy(np.ascontiguousarray(item))
    if not isinstance(item, torch.Tensor):
        raise TypeError(f"Unsupported audio input of type {type(item)}")
    item = item.to(torch.float32)
    if item.dim() == 2:  # (channels, samples) -> mono
        item = item.mean(dim=0)
    if item.dim() != 1:
        raise ValueError(f"Expected a 1-D mono waveform, got shape {tuple(item.shape)}")
    return item


def compute_fbank(
    audios: Sequence[AudioInput],
    fbank: Fbank,
    device: Union[str, torch.device] = "cpu",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute padded fbank features for a list of audio inputs.

    Returns ``(features, feature_lens)`` with shapes ``(B, T, 80)`` and ``(B,)``.
    Each item is processed separately (as in training) and the batch is padded
    with ``log(1e-10)``.
    """
    feats: List[torch.Tensor] = []
    for a in audios:
        wav = to_waveform(a, fbank.sampling_rate).to(device)
        feats.append(fbank(wav)[0])
    lens = torch.tensor([f.size(0) for f in feats], dtype=torch.long, device=device)
    x = pad_sequence(feats, batch_first=True, padding_value=LOG_EPSILON)
    return x, lens

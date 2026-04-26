"""Audio transforms used to synthesize training targets for TONE.

BrownNoiseTargetEQ: closed-form 5-band EQ that pushes a clip's log-mel
spectrum toward a -6 dB/oct (brown-noise) slope. Returns the y_target that
the auto-EQ network should learn to produce from x.

SaturatorCurveSynth: applies a fixed tube-ish soft-clip with 2nd-harmonic
asymmetry to produce (dry, wet) pairs for the Rational saturator.
"""

from __future__ import annotations

import math
from typing import Optional

import torch

from ..processors.ddsp import ParametricEQ
from ..processors.dsp import biquad, fft_sosfreqz, sosfilt_via_fsm

__all__ = ["BrownNoiseTargetEQ", "SaturatorCurveSynth"]


# TONE stage-2 frozen band centers (Hz). These match the ParametricEQ defaults
# when freeze_freqs=True; see `nablafx.processors.ddsp.ParametricEQ`.
_TONE_EQ_BAND_CENTERS_HZ = (
    ("low_shelf", 1010.0, "low_shelf"),
    ("band0", 110.0, "peaking"),
    ("band1", 1100.0, "peaking"),
    ("band2", 7000.0, "peaking"),
    ("high_shelf", 10000.0, "high_shelf"),
)
_TONE_EQ_FIXED_Q = 0.707  # broad, non-resonant


def _mel_band_log_power(
    x: torch.Tensor,
    sample_rate: float,
    n_fft: int = 2048,
    n_mels: int = 64,
    eps: float = 1e-8,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (mel_freqs_hz [n_mels], log_power [bs, n_mels]) for a clip.

    Uses the full clip (no hop) — we want a time-averaged spectral balance,
    not a spectrogram.
    """
    bs = x.shape[0]
    flat = x.reshape(bs, -1)
    window = torch.hann_window(n_fft, device=x.device, dtype=x.dtype)
    # Time-average over frames, then use the whole clip.
    spec = torch.stft(
        flat, n_fft=n_fft, hop_length=n_fft // 4, win_length=n_fft,
        window=window, return_complex=True, center=True,
    )
    power = (spec.real ** 2 + spec.imag ** 2).mean(dim=-1)  # [bs, n_freqs]
    n_freqs = power.shape[-1]
    fft_freqs = torch.linspace(0, sample_rate / 2, n_freqs, device=x.device, dtype=x.dtype)

    # Simple triangular mel filterbank.
    mel_min = 2595.0 * math.log10(1.0 + 20.0 / 700.0)
    mel_max = 2595.0 * math.log10(1.0 + (sample_rate / 2) / 700.0)
    mel_points = torch.linspace(mel_min, mel_max, n_mels + 2, device=x.device, dtype=x.dtype)
    hz_points = 700.0 * (10 ** (mel_points / 2595.0) - 1.0)
    bins = (hz_points * (n_fft / sample_rate)).clamp(0, n_freqs - 1)
    fb = torch.zeros(n_mels, n_freqs, device=x.device, dtype=x.dtype)
    for m in range(n_mels):
        left, center, right = bins[m], bins[m + 1], bins[m + 2]
        idx = torch.arange(n_freqs, device=x.device, dtype=x.dtype)
        up = (idx - left) / (center - left + eps)
        dn = (right - idx) / (right - center + eps)
        fb[m] = torch.clamp(torch.minimum(up, dn), min=0.0)

    mel_power = power @ fb.T  # [bs, n_mels]
    log_power = 10.0 * torch.log10(mel_power + eps)
    mel_hz = hz_points[1:-1]  # center of each band
    return mel_hz, log_power


def _brown_noise_target_db(mel_hz: torch.Tensor, ref_db: torch.Tensor) -> torch.Tensor:
    """Brown noise reference log-power at the mel band centers.

    -6 dB / octave below a reference frequency (1 kHz). `ref_db` is the target
    level at 1 kHz — set so the reference has the same overall energy as the
    input clip, which we do outside by matching means.
    """
    freq_ref_hz = 1000.0
    octaves = torch.log2(mel_hz.clamp(min=1.0) / freq_ref_hz)
    # shape: [n_mels]; broadcast against ref_db [bs]
    return ref_db.unsqueeze(-1) - 6.0 * octaves.unsqueeze(0)


class BrownNoiseTargetEQ(torch.nn.Module):
    """Compute y_target = EQ(x) where the EQ pushes x toward -6 dB/oct.

    For each clip we fit a closed-form 5-band gain vector (frequencies fixed)
    that minimizes mean-squared log-mel deviation from the brown-noise
    reference, apply that EQ, and return the filtered audio along with the
    gain vector that produced it.

    The gain vector is bounded to [min_gain_db, max_gain_db] per band via
    saturating clamp. This is consistent with the ParametricEQ limits in the
    TONE config.
    """

    def __init__(
        self,
        sample_rate: float,
        min_gain_db: float = -9.0,
        max_gain_db: float = 9.0,
        n_fft: int = 2048,
        n_mels: int = 64,
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.min_gain_db = min_gain_db
        self.max_gain_db = max_gain_db
        self.n_fft = n_fft
        self.n_mels = n_mels
        self.band_centers = torch.tensor(
            [c for _, c, _ in _TONE_EQ_BAND_CENTERS_HZ], dtype=torch.float32
        )
        self.band_kinds = [k for _, _, k in _TONE_EQ_BAND_CENTERS_HZ]
        self.band_names = [n for n, _, _ in _TONE_EQ_BAND_CENTERS_HZ]

    def _build_design_matrix(self, mel_hz: torch.Tensor) -> torch.Tensor:
        """A[m, k] = log-mag response (dB) of band k at +1 dB gain, Q=0.707,
        projected onto mel band m. Built once per dtype/device; cached.

        Computed from the actual biquad frequency response — not a surrogate.
        """
        cache_key = (mel_hz.device, mel_hz.dtype, int(mel_hz.shape[0]))
        cache = getattr(self, "_A_cache", {})
        if cache_key in cache:
            return cache[cache_key]

        n_fft_resp = 4096
        fft_freqs = torch.linspace(
            0, self.sample_rate / 2, n_fft_resp // 2 + 1,
            device=mel_hz.device, dtype=mel_hz.dtype,
        )
        # Map each fft bin to the nearest mel band (simple bucketing; mel_hz is
        # monotone so we can use searchsorted).
        edges = torch.cat([
            mel_hz[:1],
            0.5 * (mel_hz[:-1] + mel_hz[1:]),
            mel_hz[-1:],
        ])
        bin_to_mel = torch.searchsorted(edges, fft_freqs).clamp(0, mel_hz.shape[0] - 1)

        A = torch.zeros(mel_hz.shape[0], len(self.band_centers),
                        device=mel_hz.device, dtype=mel_hz.dtype)
        one_db = torch.tensor([1.0], device=mel_hz.device, dtype=mel_hz.dtype)
        q = torch.tensor([_TONE_EQ_FIXED_Q], device=mel_hz.device, dtype=mel_hz.dtype)
        for k, (_, center, kind) in enumerate(_TONE_EQ_BAND_CENTERS_HZ):
            center_t = torch.tensor([center], device=mel_hz.device, dtype=mel_hz.dtype)
            b, a = biquad(one_db, center_t, q, self.sample_rate, kind)
            sos = torch.cat((b, a), dim=-1).view(1, 1, 6)
            H = fft_sosfreqz(sos, n_fft=n_fft_resp)[0]  # [n_bins]
            mag_db = 20.0 * torch.log10(H.abs() + 1e-12)
            # Average mag_db within each mel bucket.
            col = torch.zeros(mel_hz.shape[0], device=mel_hz.device, dtype=mel_hz.dtype)
            counts = torch.zeros_like(col)
            col.scatter_add_(0, bin_to_mel, mag_db)
            counts.scatter_add_(0, bin_to_mel, torch.ones_like(mag_db))
            A[:, k] = col / counts.clamp(min=1.0)

        cache[cache_key] = A
        self._A_cache = cache
        return A

    @torch.no_grad()
    def _solve_gains(self, x: torch.Tensor) -> torch.Tensor:
        """Return per-band gain_db tensor [bs, 5]."""
        mel_hz, log_power = _mel_band_log_power(
            x, self.sample_rate, n_fft=self.n_fft, n_mels=self.n_mels
        )
        # Reference level at 1 kHz: match the input clip's overall log-mean.
        ref_db = log_power.mean(dim=-1)
        target_db = _brown_noise_target_db(mel_hz, ref_db)
        delta_db = target_db - log_power  # [bs, n_mels]

        A = self._build_design_matrix(mel_hz)  # [n_mels, 5], dB per +1 dB gain
        # Solve ridge-regularized least squares A g ≈ delta_db.
        At = A.T
        AtA = At @ A + 1e-2 * torch.eye(
            len(self.band_centers), device=x.device, dtype=x.dtype
        )
        rhs = (At @ delta_db.T).T  # [bs, 5]
        g = torch.linalg.solve(AtA, rhs.unsqueeze(-1)).squeeze(-1)
        return g.clamp(self.min_gain_db, self.max_gain_db)

    def _apply_eq(self, x: torch.Tensor, gains_db: torch.Tensor) -> torch.Tensor:
        """Run x through a 5-SOS biquad chain with the given per-band gains."""
        bs, chs, _ = x.shape
        assert chs == 1
        sos = torch.zeros(bs, 5, 6, device=x.device, dtype=x.dtype)
        for k, (name, center, kind) in enumerate(_TONE_EQ_BAND_CENTERS_HZ):
            gain_k = gains_db[:, k]
            center_k = torch.full_like(gain_k, center)
            q_k = torch.full_like(gain_k, _TONE_EQ_FIXED_Q)
            b, a = biquad(gain_k, center_k, q_k, self.sample_rate, kind)
            sos[:, k, :] = torch.cat((b, a), dim=-1)
        return sosfilt_via_fsm(sos, x)

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (y_target, gains_db [bs, 5]) for input x [bs, 1, T]."""
        if x.dim() == 2:
            x = x.unsqueeze(1)
        gains_db = self._solve_gains(x)
        y = self._apply_eq(x, gains_db)
        return y, gains_db


class SaturatorCurveSynth(torch.nn.Module):
    """Apply a fixed soft-clip curve with 2nd-harmonic asymmetry.

        y = sign(x) * (1 - exp(-k * |x|)) + alpha * x^2 * sign(x)

    Defaults produce a gentle tube-ish curve. Inputs are expected at roughly
    line level (RMS ~ -18 dBFS). The Rational saturator is trained to reproduce
    this curve at unity drive; the Amount knob modulates pre-/post-gain in the
    plugin host.
    """

    def __init__(self, k: float = 3.5, alpha: float = 0.1):
        super().__init__()
        self.k = k
        self.alpha = alpha

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        sx = torch.sign(x)
        return sx * (1.0 - torch.exp(-self.k * x.abs())) + self.alpha * x.pow(2) * sx

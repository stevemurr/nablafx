"""Generate the synthetic saturator dataset for TONE stage 3a.

Emits (dry, wet) pairs where:
- dry is a mix of sines, chirps, noise, and harmonic content at varied levels.
- wet is dry passed through SaturatorCurveSynth (fixed tube-ish soft clip).

Training target: the Rational nonlinearity reproduces the curve at unity drive
(peak around ±1.0). The plugin host modulates pre/post-gain for the Amount knob.

Layout matches PluginDataset expectations:
  <out>/DRY/{split}/{name}.input.wav
  <out>/WET/{split}/{name}.target.wav

Usage:
  uv run python scripts/generate_sat_dataset.py \
      --out /shared/datasets/tone_sat \
      --sample-rate 44100 --duration 3.0 \
      --num-trainval 200 --num-test 20
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from nablafx.data.transforms import SaturatorCurveSynth


def _synthesize_clip(rng: np.random.Generator, sr: int, duration_s: float) -> np.ndarray:
    """One clip: random mix of tones, chirps, filtered noise at varying level."""
    n = int(sr * duration_s)
    t = np.arange(n, dtype=np.float32) / sr
    out = np.zeros(n, dtype=np.float32)

    # 1–3 tones at random frequencies with random phases/amplitudes.
    n_tones = rng.integers(1, 4)
    for _ in range(int(n_tones)):
        f = float(rng.uniform(80.0, 4000.0))
        phase = float(rng.uniform(0.0, 2 * np.pi))
        amp = float(rng.uniform(0.1, 0.5))
        out += amp * np.sin(2 * np.pi * f * t + phase).astype(np.float32)

    # Optional chirp (exercises the curve across frequency).
    if rng.random() < 0.5:
        f0 = float(rng.uniform(50.0, 500.0))
        f1 = float(rng.uniform(1000.0, 8000.0))
        k = (f1 / f0) ** (1.0 / duration_s)
        phase_inst = 2 * np.pi * f0 * ((k**t - 1.0) / np.log(k + 1e-9))
        out += float(rng.uniform(0.05, 0.3)) * np.sin(phase_inst).astype(np.float32)

    # Colored noise (pink-ish via cumsum then lowpass-ish).
    if rng.random() < 0.7:
        noise = rng.standard_normal(n).astype(np.float32)
        # cheap 1st-order lowpass
        a = rng.uniform(0.8, 0.99)
        y = np.zeros_like(noise)
        acc = 0.0
        for i in range(n):
            acc = float(a) * acc + (1.0 - float(a)) * noise[i]
            y[i] = acc
        out += float(rng.uniform(0.05, 0.3)) * y

    # Randomly scale peak into [0.3, 1.5] so the curve sees both clean and
    # heavily-driven regions (post-clip the Rational will learn the shape).
    peak = float(np.max(np.abs(out)) + 1e-8)
    target_peak = float(rng.uniform(0.3, 1.5))
    out = out * (target_peak / peak)
    return out.astype(np.float32)


def _write(path: Path, audio: np.ndarray, sr: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, audio, sr, subtype="PCM_24")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="Output root")
    ap.add_argument("--sample-rate", type=int, default=44100)
    ap.add_argument("--duration", type=float, default=3.0)
    ap.add_argument("--num-trainval", type=int, default=200)
    ap.add_argument("--num-test", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    out_root = Path(args.out)
    rng = np.random.default_rng(args.seed)
    sat = SaturatorCurveSynth()

    for split, count in [("trainval", args.num_trainval), ("test", args.num_test)]:
        dry_dir = out_root / "DRY" / split
        wet_dir = out_root / "WET" / split
        for i in range(count):
            dry = _synthesize_clip(rng, args.sample_rate, args.duration)
            dry_t = torch.from_numpy(dry).view(1, 1, -1)
            wet_t = sat(dry_t).view(-1).numpy()
            name = f"sat_{i:05d}"
            _write(dry_dir / f"{name}.input.wav", dry, args.sample_rate)
            _write(wet_dir / f"{name}.target.wav", wet_t.astype(np.float32), args.sample_rate)
        print(f"{split}: wrote {count} pairs to {out_root}")


if __name__ == "__main__":
    main()

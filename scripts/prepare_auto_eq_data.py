"""Prepare the auto-EQ training dataset for TONE stage 2.

Reads clean broadband audio (music, ideally — MUSDB18 mixtures, or any
directory of .wav files) and writes paired (dry, wet) clips where the wet
version is the result of applying BrownNoiseTargetEQ to the dry. The network
then learns to reproduce that EQ directly from the dry signal.

Output layout matches PluginDataset:
  <out>/DRY/{split}/{name}.input.wav
  <out>/WET/{split}/{name}.target.wav

Usage (synth stand-in corpus while we don't have MUSDB18):
  uv run python scripts/prepare_auto_eq_data.py \\
      --synth --out /shared/datasets/tone_auto_eq \\
      --num-trainval 300 --num-test 30

Usage (real music corpus):
  uv run python scripts/prepare_auto_eq_data.py \\
      --src /path/to/MUSDB18/mixtures \\
      --out /shared/datasets/tone_auto_eq \\
      --clip-seconds 3.0 --num-trainval 400 --num-test 40
"""

from __future__ import annotations

import argparse
import glob
import os
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from nablafx.data.transforms import BrownNoiseTargetEQ


def _write(path: Path, audio: np.ndarray, sr: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, audio, sr, subtype="PCM_24")


def _synth_music_like_clip(rng: np.random.Generator, sr: int, seconds: float) -> np.ndarray:
    """Synthesize a broadband 'music-like' clip: drone + harmonics + noise bed
    + occasional transients. Useful as a stand-in when we don't have MUSDB18
    available for smoke-testing the auto-EQ loop.
    """
    n = int(sr * seconds)
    t = np.arange(n, dtype=np.float32) / sr
    out = np.zeros(n, dtype=np.float32)

    # Chord: random root + 2-4 overtones.
    root = float(rng.uniform(80.0, 300.0))
    n_partials = rng.integers(3, 6)
    for k in range(int(n_partials)):
        f = root * (k + 1) * float(rng.uniform(0.98, 1.02))
        amp = 0.3 / (k + 1) * float(rng.uniform(0.6, 1.2))
        phase = float(rng.uniform(0, 2 * np.pi))
        out += amp * np.sin(2 * np.pi * f * t + phase).astype(np.float32)

    # Noise bed (broadband).
    bed = rng.standard_normal(n).astype(np.float32) * 0.1
    out += bed

    # Occasional transient ticks (drum-like energy across the spectrum).
    for _ in range(int(rng.integers(0, 5))):
        pos = int(rng.uniform(0.1, 0.9) * n)
        length = int(0.02 * sr)
        env = np.exp(-np.arange(length) / (0.006 * sr)).astype(np.float32)
        tick = rng.standard_normal(length).astype(np.float32) * env * 0.5
        end = min(pos + length, n)
        out[pos:end] += tick[: end - pos]

    # Random overall tilt so targets have variety: apply ±6 dB/oct log-freq
    # tilt in the time domain via a one-pole LP or HP emphasis.
    tilt_oct = float(rng.uniform(-3.0, +3.0))
    if abs(tilt_oct) > 0.1:
        alpha = np.clip(np.exp(-abs(tilt_oct) * 0.2), 0.5, 0.99)
        y = np.zeros_like(out)
        acc = 0.0
        for i in range(n):
            acc = alpha * acc + (1.0 - alpha) * out[i]
            y[i] = out[i] - acc if tilt_oct > 0 else acc
        out = y.astype(np.float32)

    # Normalize peak to ~0.5 so the EQ has headroom.
    peak = float(np.max(np.abs(out)) + 1e-8)
    out *= 0.5 / peak
    return out


def _crop_random(x: np.ndarray, want: int, rng: np.random.Generator) -> np.ndarray:
    if x.ndim > 1:
        x = x.mean(axis=-1)  # downmix to mono
    if x.shape[0] < want:
        # Pad with zeros to reach the requested length.
        pad = want - x.shape[0]
        return np.concatenate([x, np.zeros(pad, dtype=x.dtype)])
    start = rng.integers(0, x.shape[0] - want + 1)
    return x[start : start + want]


def _load_audio(path: str, sr_target: int) -> np.ndarray | None:
    try:
        x, sr = sf.read(path, dtype="float32")
    except Exception:
        return None
    if sr != sr_target:
        # sf.read doesn't resample; cheap nearest-neighbor would be wrong.
        # For now we require input audio to already be at sr_target.
        print(f"  skipping {path}: sr={sr} != {sr_target}")
        return None
    return x


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="Output root")
    ap.add_argument("--synth", action="store_true",
                    help="Generate synthetic broadband clips instead of reading --src")
    ap.add_argument("--src", default=None, help="Source directory of .wav files (recurses)")
    ap.add_argument("--sample-rate", type=int, default=44100)
    ap.add_argument("--clip-seconds", type=float, default=3.0)
    ap.add_argument("--num-trainval", type=int, default=300)
    ap.add_argument("--num-test", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    out_root = Path(args.out)
    rng = np.random.default_rng(args.seed)
    sr = args.sample_rate
    clip_samples = int(args.clip_seconds * sr)

    # If reading real audio, gather file list.
    src_files: list[str] = []
    if not args.synth:
        if args.src is None:
            raise SystemExit("--src required when --synth is not set")
        for ext in ("*.wav", "*.flac"):
            src_files.extend(glob.glob(os.path.join(args.src, "**", ext), recursive=True))
        if not src_files:
            raise SystemExit(f"no audio files found under {args.src}")
        print(f"found {len(src_files)} source files")

    # Target EQ solver — runs on CPU, batches don't help for the single-clip
    # inner call, so do them one-by-one.
    eq = BrownNoiseTargetEQ(sample_rate=sr)

    for split, count in [("trainval", args.num_trainval), ("test", args.num_test)]:
        dry_dir = out_root / "DRY" / split
        wet_dir = out_root / "WET" / split
        for i in range(count):
            if args.synth:
                dry = _synth_music_like_clip(rng, sr, args.clip_seconds)
            else:
                path = src_files[int(rng.integers(0, len(src_files)))]
                loaded = _load_audio(path, sr)
                if loaded is None:
                    # try another
                    continue
                dry = _crop_random(loaded, clip_samples, rng).astype(np.float32)

            # Compute wet (brown-noise-tilted).
            x_t = torch.from_numpy(dry).view(1, 1, -1)
            wet_t, gains_db = eq(x_t)
            wet = wet_t.view(-1).numpy().astype(np.float32)

            name = f"ae_{i:05d}"
            _write(dry_dir / f"{name}.input.wav", dry, sr)
            _write(wet_dir / f"{name}.target.wav", wet, sr)

            if (i + 1) % 50 == 0:
                print(f"  {split}: {i+1}/{count}  gains_db={gains_db[0].tolist()}")

        print(f"{split}: wrote {count} pairs to {out_root}")


if __name__ == "__main__":
    main()

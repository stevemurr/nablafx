#!/usr/bin/env python3
"""Reshape the SignalTrain LA2A dataset into the nablafx parametric layout.

SignalTrain ships files like:
    <src>/Train/input_138_.wav
    <src>/Train/target_138_LA2A_2c__1__65.wav   # switch=1, peak_red=65

nablafx ParametricPluginDataset expects:
    <dst>/DRY/trainval/138.input.wav
    <dst>/LA2A/trainval/C100_P065/C100_P065.138.target.wav

Params use letter-value tokens (`ParametricPluginDataset` parses each one as
``float(value)/100``), so we encode:
  - Comp/Limit switch:  C000 or C100   (yields 0.0 or 1.0 after normalization)
  - Peak Reduction:     P{0..100}      (yields 0.0 to 1.0 after normalization)

By default the converter symlinks so the ~few-GB audio isn't duplicated.
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import sys
from collections import defaultdict
from pathlib import Path


_SIGNALTRAIN_SPLITS = {"Train": "trainval", "Val": "trainval", "Test": "test"}
_WET_SUBDIR = "LA2A-CompLimiter"

_INPUT_RE  = re.compile(r"^input_(\d+)_\.wav$")
# SignalTrain uses both "LA2A_2c" and "LA2A_3c" prefixes across splits — same
# format otherwise. Match either.
_TARGET_RE = re.compile(r"^target_(\d+)_LA2A_\dc__(\d+)__(\d+)\.wav$")


def _params_token(switch: int, peak_reduction: int) -> str:
    # Comp/Limit is boolean; scale to {0, 100} so the /100 normalization gives
    # {0.0, 1.0}. Peak reduction is already 0..100.
    return f"C{switch * 100:03d}_P{peak_reduction:03d}"


def _link(src: Path, dst: Path, copy: bool) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if copy:
        shutil.copy2(src, dst)
    else:
        # Relative symlink so the same dataset tree works across bind-mounts
        # (e.g. /home/murr/... on the host, /shared/datasets/... in a container).
        rel = os.path.relpath(src, dst.parent)
        os.symlink(rel, dst)


def convert(src_root: Path, dst_root: Path, copy: bool = False) -> None:
    if not src_root.is_dir():
        raise SystemExit(f"error: src {src_root} is not a directory")

    # Fail loud if IDs would collide when we merge Train+Val under trainval/.
    ids_per_split: dict[str, set[str]] = defaultdict(set)

    n_inputs = 0
    n_targets = 0

    for src_split, dst_split in _SIGNALTRAIN_SPLITS.items():
        src_dir = src_root / src_split
        if not src_dir.is_dir():
            print(f"  note: {src_dir} not present, skipping", file=sys.stderr)
            continue

        for entry in sorted(src_dir.iterdir()):
            name = entry.name
            m_in = _INPUT_RE.match(name)
            if m_in:
                file_id = m_in.group(1)
                if file_id in ids_per_split[dst_split]:
                    raise SystemExit(
                        f"id collision: {file_id} appears twice under {dst_split}. "
                        "Inspect the SignalTrain splits before re-running."
                    )
                ids_per_split[dst_split].add(file_id)
                dst = dst_root / "DRY" / dst_split / f"{file_id}.input.wav"
                _link(entry, dst, copy)
                n_inputs += 1
                continue

            m_tgt = _TARGET_RE.match(name)
            if m_tgt:
                file_id = m_tgt.group(1)
                switch = int(m_tgt.group(2))
                peak_red = int(m_tgt.group(3))
                params = _params_token(switch, peak_red)
                dst = (
                    dst_root
                    / _WET_SUBDIR
                    / dst_split
                    / params
                    / f"{params}.{file_id}.target.wav"
                )
                _link(entry, dst, copy)
                n_targets += 1
                continue

    print(f"converted {n_inputs} inputs, {n_targets} targets into {dst_root}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Reshape SignalTrain LA2A into nablafx ParametricPluginDataset layout",
    )
    parser.add_argument("--src", required=True, type=Path, help="SignalTrain dataset root")
    parser.add_argument("--dst", required=True, type=Path, help="Output dataset root")
    parser.add_argument(
        "--copy",
        action="store_true",
        help="Copy files instead of symlinking (default is symlink)",
    )
    args = parser.parse_args(argv)
    convert(args.src.resolve(), args.dst.resolve(), copy=args.copy)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Build the composite TONE staging bundle.

Two modes:

  Bundles already exported (the typical Mac-clone-and-build path)::

      python scripts/export_tone.py from-bundles \\
          --auto-eq-bundle   artifacts/tone-bundles/auto_eq \\
          --saturator-bundle artifacts/tone-bundles/saturator \\
          --la2a-bundle      artifacts/tone-bundles/la2a \\
          --out              build/tone-staging

  From-scratch (export each stage from a Hydra run dir, then compose)::

      python scripts/export_tone.py from-runs \\
          --auto-eq-run   /shared/artifacts/auto_eq_brown/outputs/2026-04-25/12-46-28 \\
          --auto-eq-ckpt  ...checkpoints/epoch=200-step=5000.ckpt \\
          --saturator-run /shared/artifacts/saturator_synth/outputs/2026-04-25/10-32-20 \\
          --saturator-ckpt ...checkpoints/epoch=500-step=4000.ckpt \\
          --la2a-run      /shared/artifacts/la2a/outputs/2026-04-24/20-10-16 \\
          --la2a-ckpt     ...checkpoints/epoch=8-step=89600.ckpt \\
          --out           build/tone-staging

The ``from-runs`` mode shells out to ``nablafx-export`` for each stage so the
exact same code path runs as you'd get from manual exports.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import nablafx  # noqa: F401 — applies rational-activations patch
from nablafx.export.composite import export_composite_bundle


def _from_bundles(args) -> int:
    meta = export_composite_bundle(
        auto_eq_bundle=Path(args.auto_eq_bundle),
        saturator_bundle=Path(args.saturator_bundle),
        la2a_bundle=Path(args.la2a_bundle),
        out_dir=Path(args.out),
        effect_name=args.effect_name,
    )
    print(f"composite bundle written to {args.out}")
    print(f"  effect_name: {meta.effect_name}")
    print(f"  model_id:    {meta.model_id}")
    print(f"  sample_rate: {meta.sample_rate}")
    return 0


def _run(cmd: list[str]) -> None:
    print(f"$ {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True)


def _from_runs(args) -> int:
    work = Path(tempfile.mkdtemp(prefix="tone-export-"))
    try:
        autoeq_dir = work / "auto_eq"
        sat_dir    = work / "saturator"
        la2a_dir   = work / "la2a"

        # nablafx-export each stage. We re-shell so any future flags added to
        # the CLI (e.g. --effect/--letters) work without duplication here.
        _run([
            "nablafx-export", "--run-dir", args.auto_eq_run,
            *(["--ckpt", args.auto_eq_ckpt] if args.auto_eq_ckpt else []),
            "--out", str(autoeq_dir),
        ])
        _run([
            "nablafx-export", "--run-dir", args.saturator_run,
            *(["--ckpt", args.saturator_ckpt] if args.saturator_ckpt else []),
            "--out", str(sat_dir),
        ])
        _run([
            "nablafx-export", "--run-dir", args.la2a_run,
            *(["--ckpt", args.la2a_ckpt] if args.la2a_ckpt else []),
            "--effect", "LA2A", "--letters", "C,P",
            "--out", str(la2a_dir),
        ])

        meta = export_composite_bundle(
            auto_eq_bundle=autoeq_dir,
            saturator_bundle=sat_dir,
            la2a_bundle=la2a_dir,
            out_dir=Path(args.out),
            effect_name=args.effect_name,
        )
        print(f"composite bundle written to {args.out}")
        print(f"  effect_name: {meta.effect_name}")
        print(f"  model_id:    {meta.model_id}")
        print(f"  sample_rate: {meta.sample_rate}")
        return 0
    finally:
        shutil.rmtree(work, ignore_errors=True)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="export_tone",
                                description="Build composite TONE staging bundle.")
    p.add_argument("--effect-name", default="TONE")
    sp = p.add_subparsers(dest="cmd", required=True)

    pb = sp.add_parser("from-bundles", help="compose three already-exported per-stage bundles")
    pb.add_argument("--auto-eq-bundle",   required=True)
    pb.add_argument("--saturator-bundle", required=True)
    pb.add_argument("--la2a-bundle",      required=True)
    pb.add_argument("--out",              required=True)
    pb.set_defaults(func=_from_bundles)

    pr = sp.add_parser("from-runs", help="export each stage from its Hydra run dir, then compose")
    pr.add_argument("--auto-eq-run",   required=True)
    pr.add_argument("--auto-eq-ckpt",  default=None)
    pr.add_argument("--saturator-run", required=True)
    pr.add_argument("--saturator-ckpt", default=None)
    pr.add_argument("--la2a-run",      required=True)
    pr.add_argument("--la2a-ckpt",     default=None)
    pr.add_argument("--out",           required=True)
    pr.set_defaults(func=_from_runs)

    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())

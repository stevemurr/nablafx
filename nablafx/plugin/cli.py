"""Console entrypoint: ``nablafx-plugin``.

Bridges the portable Python export step and the mac-native CLAP build step:

  - Runs ``nablafx.export.export_bundle`` to produce a staging directory.
  - If we're on macOS, invokes ``native/clap/build.sh`` to turn that
    staging directory into a code-signed ``.clap`` bundle.
  - Otherwise, prints the staging directory and instructions to rsync it to
    a Mac and run ``build.sh`` there.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import yaml

import nablafx  # noqa: F401

from nablafx.export.bundle import ExportInputs, export_bundle
from nablafx.export.validate import ExportValidationError


_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_CONF_EFFECT_DIR = _REPO_ROOT / "conf" / "effect"
_NATIVE_BUILD_SH = _REPO_ROOT / "native" / "clap" / "build.sh"


def _load_effect_cfg(name: str | None) -> dict | None:
    if not name:
        return None
    path = _CONF_EFFECT_DIR / f"{name}.yaml"
    if not path.is_file():
        raise FileNotFoundError(
            f"Effect config not found at {path}. Pick one of: "
            f"{sorted(p.stem for p in _CONF_EFFECT_DIR.glob('*.yaml'))}"
        )
    return yaml.safe_load(path.read_text())


def _parse_letters(letters_arg: str | None) -> list[str] | None:
    if letters_arg is None:
        return None
    return [p.strip() for p in letters_arg.split(",") if p.strip()]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="nablafx-plugin",
        description=(
            "Export a trained nablafx checkpoint and build a CLAP plugin "
            "bundle (macOS arm64). On non-mac hosts this stops after export "
            "and prints scp instructions."
        ),
    )
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument(
        "--out",
        required=True,
        type=Path,
        help="Output .clap bundle path (or staging directory on non-mac hosts)",
    )
    parser.add_argument("--effect", default=None)
    parser.add_argument("--letters", default=None)
    parser.add_argument(
        "--staging",
        default=None,
        type=Path,
        help="Optional explicit staging directory; defaults to <out>.staging",
    )
    args = parser.parse_args(argv)

    staging_dir = args.staging or Path(str(args.out) + ".staging")
    try:
        effect_cfg = _load_effect_cfg(args.effect)
        meta = export_bundle(
            ExportInputs(
                run_dir=args.run_dir,
                out_dir=staging_dir,
                effect_name=(effect_cfg.get("name") if effect_cfg else None),
                effect_cfg=effect_cfg,
                letters_in_use=_parse_letters(args.letters),
            )
        )
    except (ExportValidationError, FileNotFoundError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if sys.platform != "darwin":
        print(
            f"\nStaging bundle ready at {staging_dir}\n"
            f"  model_id: {meta.model_id}\n"
            f"\nTo finish the build, copy the staging dir plus native/clap/ to a\n"
            f"macOS arm64 machine and run:\n\n"
            f"  cd native/clap && ./build.sh <staging> {args.out}\n"
        )
        return 0

    if not _NATIVE_BUILD_SH.is_file():
        print(f"error: native build script not found at {_NATIVE_BUILD_SH}", file=sys.stderr)
        return 2

    proc = subprocess.run(
        [str(_NATIVE_BUILD_SH), str(staging_dir), str(args.out)],
        check=False,
    )
    return proc.returncode


if __name__ == "__main__":
    raise SystemExit(main())

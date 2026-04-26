"""Console entrypoint: ``nablafx-export``.

Example (non-parametric 1176 limiter):

    nablafx-export --run-dir outputs/2026-04-22/19-21-19 \
                   --out /tmp/1176LN-staging

Example (parametric):

    nablafx-export --run-dir <run> --effect UA-1176LN --out <staging>

The ``--effect`` argument is the stem of a file under ``conf/effect/``. It's
only required for parametric models (num_controls > 0).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

import nablafx  # noqa: F401 — applies rational-activations patch

from nablafx.core.models import BlackBoxModel, GreyBoxModel

from .bundle import ExportInputs, _load_system_and_weights, export_bundle
from .grey_bundle import export_grey_bundle
from .validate import ExportValidationError


_CONF_EFFECT_DIR = Path(__file__).resolve().parent.parent.parent / "conf" / "effect"


def _parse_letters(letters_arg: str | None) -> list[str] | None:
    if letters_arg is None:
        return None
    return [p.strip() for p in letters_arg.split(",") if p.strip()]


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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="nablafx-export",
        description="Export a trained nablafx checkpoint to a CLAP plugin staging bundle.",
    )
    parser.add_argument(
        "--run-dir",
        required=True,
        type=Path,
        help="Hydra output directory (must contain .hydra/config.yaml and checkpoints/last.ckpt)",
    )
    parser.add_argument(
        "--out",
        required=True,
        type=Path,
        help="Staging directory to write (model.onnx, plugin_meta.json, source.hydra.yaml)",
    )
    parser.add_argument(
        "--effect",
        default=None,
        help="Effect id (stem of conf/effect/<name>.yaml). Required if the model has controls.",
    )
    parser.add_argument(
        "--letters",
        default=None,
        help=(
            "Comma-separated letters for the model's control inputs, in the "
            "order the model sees them. Required for parametric models. "
            "Example: --letters A,R,I,O"
        ),
    )
    parser.add_argument(
        "--ckpt",
        type=Path,
        default=None,
        help=(
            "Specific .ckpt under run-dir to export. Defaults to "
            "checkpoints/last.ckpt. Pass the best-by-val ckpt "
            "(e.g. checkpoints/epoch=N-step=M.ckpt) to ship the converged "
            "weights for runs stopped past their plateau."
        ),
    )
    args = parser.parse_args(argv)

    try:
        effect_cfg = _load_effect_cfg(args.effect)
        export_inputs = ExportInputs(
            run_dir=args.run_dir,
            out_dir=args.out,
            effect_name=(effect_cfg.get("name") if effect_cfg else None),
            effect_cfg=effect_cfg,
            letters_in_use=_parse_letters(args.letters),
            ckpt_path=args.ckpt,
        )
        # Dispatch on the trained system's model class so callers don't have to
        # know which export pipeline to invoke.
        system = _load_system_and_weights(args.run_dir, ckpt_path=args.ckpt)
        if isinstance(system.model, GreyBoxModel):
            meta = export_grey_bundle(export_inputs)
        elif isinstance(system.model, BlackBoxModel):
            meta = export_bundle(export_inputs)
        else:
            raise ExportValidationError(
                f"Unsupported model class {type(system.model).__name__}"
            )
    except ExportValidationError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    print(f"wrote bundle to {args.out}")
    print(f"  stage_kind:       {meta.stage_kind}")
    print(f"  architecture:     {meta.architecture}")
    print(f"  model_id:         {meta.model_id}")
    print(f"  sample_rate:      {meta.sample_rate}")
    print(f"  receptive_field:  {meta.receptive_field}")
    print(f"  causal:           {meta.causal}")
    print(f"  num_controls:     {meta.num_controls}")
    print(f"  state_tensors:    {len(meta.state_tensors)}")
    print(f"  dsp_blocks:       {len(meta.dsp_blocks)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

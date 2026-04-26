"""Composite TONE plugin export.

Reads three already-built per-stage bundles (la2a, saturator, auto_eq —
produced by ``nablafx-export``) and emits a single staging directory ready
for ``native/clap/build.sh tone`` on macOS.

Composite layout written to ``out_dir``::

    tone_meta.json            # this module's schema; see CompositePluginMeta
    la2a/         (copied from input bundle, model.onnx + plugin_meta.json + source.hydra.yaml)
    saturator/    (no model.onnx — pure DSP stage)
    auto_eq/      (model.onnx is the controller LSTM; DSP block in plugin_meta.json)

The C++ side (``native/clap/src/composite_meta.cpp``) loads ``tone_meta.json``
and the three sub-``plugin_meta.json`` files to build the runtime chain.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


SCHEMA_VERSION = 1


@dataclass(frozen=True)
class _AmountMappingSat:
    pre_gain_db_max:  float = 12.0
    post_gain_db_max: float = -12.0
    wet_mix_max:      float = 1.0


@dataclass(frozen=True)
class _AmountMappingLa2a:
    peak_reduction_min: float = 20.0
    peak_reduction_max: float = 70.0
    comp_or_limit:      float = 1.0   # held at "Limit" in the composite spec


@dataclass(frozen=True)
class _AmountMappingAutoEq:
    wet_mix_max: float = 1.0


@dataclass(frozen=True)
class CompositePluginMeta:
    """Top-level meta for the composite TONE plugin.

    The C++ host reads this once at module load to wire AMT → per-stage params,
    locate sub-bundles, and configure the in-host DSP stages (LUFS leveler,
    true-peak ceiling, output trim).
    """
    schema_version: int = SCHEMA_VERSION
    effect_name:    str = "TONE"
    model_id:       str = ""
    sample_rate:    int = 44100
    channels:       int = 1
    # Sub-bundles, by role name. Each entry's ``bundle_dir`` is a directory
    # name relative to the staging dir / the .clap Resources dir.
    sub_bundles:    Dict[str, str] = field(default_factory=dict)
    # AMT (Amount) and TRM (Output Trim) — host-exposed knobs.
    controls:       Dict[str, Dict[str, Any]] = field(default_factory=dict)
    # Per-stage mapping from AMT ∈ [0, 1] to internal parameters.
    amount_mapping: Dict[str, Dict[str, float]] = field(default_factory=dict)
    leveler:        Dict[str, float] = field(default_factory=dict)
    ceiling:        Dict[str, float] = field(default_factory=dict)


def _build_default_meta(model_id: str, sample_rate: int) -> CompositePluginMeta:
    return CompositePluginMeta(
        model_id=model_id,
        sample_rate=sample_rate,
        sub_bundles={
            "auto_eq":   "auto_eq",
            "saturator": "saturator",
            "la2a":      "la2a",
        },
        controls={
            "AMT": {"id": "AMT", "name": "Amount",      "min": 0.0,   "max": 1.0,
                    "default": 0.5, "skew": 1.0, "unit": ""},
            "TRM": {"id": "TRM", "name": "Output Trim", "min": -12.0, "max": 12.0,
                    "default": 0.0, "skew": 1.0, "unit": "dB"},
        },
        amount_mapping={
            "saturator": asdict(_AmountMappingSat()),
            "la2a":      asdict(_AmountMappingLa2a()),
            "auto_eq":   asdict(_AmountMappingAutoEq()),
        },
        leveler={"target_lufs": -14.0},
        ceiling={"ceiling_dbtp": -1.0, "lookahead_ms": 1.5,
                 "attack_ms": 0.5, "release_ms": 50.0},
    )


def _load_sub_meta(bundle_dir: Path) -> Dict[str, Any]:
    p = bundle_dir / "plugin_meta.json"
    if not p.is_file():
        raise FileNotFoundError(f"missing {p}")
    return json.loads(p.read_text())


def _check_sub_bundle(bundle_dir: Path, expected_kind: str, expected_block_kind: Optional[str] = None) -> Dict[str, Any]:
    meta = _load_sub_meta(bundle_dir)
    sk = meta.get("stage_kind")
    if sk != expected_kind:
        raise ValueError(f"{bundle_dir}: expected stage_kind={expected_kind!r}, got {sk!r}")
    if expected_block_kind:
        blocks = meta.get("dsp_blocks") or []
        if not blocks or blocks[0].get("kind") != expected_block_kind:
            raise ValueError(
                f"{bundle_dir}: expected dsp_blocks[0].kind={expected_block_kind!r}, "
                f"got {[b.get('kind') for b in blocks]}"
            )
    return meta


def export_composite_bundle(
    auto_eq_bundle: Path,
    saturator_bundle: Path,
    la2a_bundle: Path,
    out_dir: Path,
    effect_name: str = "TONE",
) -> CompositePluginMeta:
    """Validate the three sub-bundles, copy them under ``out_dir``, and write
    the composite ``tone_meta.json``."""
    auto_eq_bundle   = Path(auto_eq_bundle).resolve()
    saturator_bundle = Path(saturator_bundle).resolve()
    la2a_bundle      = Path(la2a_bundle).resolve()
    out_dir = Path(out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    autoeq_meta = _check_sub_bundle(auto_eq_bundle,   expected_kind="nn+dsp",
                                    expected_block_kind="parametric_eq_5band")
    sat_meta    = _check_sub_bundle(saturator_bundle, expected_kind="dsp",
                                    expected_block_kind="rational_a")
    la2a_meta   = _check_sub_bundle(la2a_bundle,      expected_kind="nn")

    sample_rates = {autoeq_meta["sample_rate"], sat_meta["sample_rate"], la2a_meta["sample_rate"]}
    if len(sample_rates) != 1:
        raise ValueError(
            f"sub-bundles disagree on sample_rate: "
            f"auto_eq={autoeq_meta['sample_rate']}, saturator={sat_meta['sample_rate']}, "
            f"la2a={la2a_meta['sample_rate']}"
        )
    sample_rate = sample_rates.pop()

    # Compose a stable model_id from the three sub-IDs so DAWs persist
    # automation against this exact model combination.
    model_id = (
        f"tone__{la2a_meta['model_id']}__{sat_meta['model_id']}__{autoeq_meta['model_id']}"
    )

    # Copy each sub-bundle dir into the composite out_dir under the stable
    # role name. Subsequent rebuilds overwrite cleanly.
    for role, src in (("auto_eq", auto_eq_bundle),
                      ("saturator", saturator_bundle),
                      ("la2a", la2a_bundle)):
        dst = out_dir / role
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(src, dst)

    meta = _build_default_meta(model_id=model_id, sample_rate=int(sample_rate))
    # Override effect_name if caller asked.
    meta = CompositePluginMeta(
        **{**asdict(meta), "effect_name": effect_name},
    )
    (out_dir / "tone_meta.json").write_text(json.dumps(asdict(meta), indent=2) + "\n")
    return meta

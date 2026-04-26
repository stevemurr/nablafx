"""Split-export for ``GreyBoxModel`` chains.

For grey-box models the runtime DSP lives in the CLAP host (see
``native/clap/src``). When the chain has only learned DSP blocks (e.g.
StaticRationalNonlinearity, no controller NN), the bundle is pure JSON. When
the chain has a learned controller predicting block-rate parameters
(e.g. ParametricEQ + DynamicController), the controller is exported to ONNX
and the DSP block layout is serialized to JSON; the C++ host runs ORT once
per block, denormalizes the predicted params, and runs the native filter
cascade.

Per processor kind the params payload is:
  StaticRationalNonlinearity → ``{"version": "A", "numerator": [...], "denominator": [...]}``
  ParametricEQ               → 5-band layout with fixed center freqs + ranges.

Adding a new kind: implement an extractor here, whitelist the class in
``validate._GREY_SUPPORTED_PROCESSORS``, and add a matching native block in
``native/clap``.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import List

import torch
from omegaconf import DictConfig, OmegaConf

from nablafx.controllers.controllers import DummyController
from nablafx.core.models import GreyBoxModel
from .bundle import (
    ExportInputs,
    _load_system_and_weights,
    _model_id_from_run,
)
from .meta import DspBlockSpec, PluginMeta, StateSpec, latency_for
from .validate import ExportValidationError, validate_grey_exportable
from .wrapper import StatefulEntry, collect_new_states, collect_stateful, seed_states


def _extract_rational_a(processor: torch.nn.Module, qname: str) -> DspBlockSpec:
    """Pull numerator + denominator weights out of a StaticRationalNonlinearity.

    The Rational version A formula is
        P(x) / Q(x) = a_0 + a_1*x + ... + a_n*x^n
                      / (1 + |b_1*x| + |b_2*x^2| + ... + |b_m*x^m|)
    so we serialize ``numerator`` (length n+1) and ``denominator`` (length m).
    """
    net = processor.net  # rational.torch.Rational
    version = getattr(net, "version", None)
    if version != "A":
        raise ExportValidationError(
            f"{qname}: only Rational version 'A' supported; got version={version!r}"
        )
    return DspBlockSpec(
        kind="rational_a",
        name=qname,
        params={
            "version": "A",
            "numerator": net.numerator.detach().cpu().tolist(),
            "denominator": net.denominator.detach().cpu().tolist(),
        },
    )


# Order matches ParametricEQ.get_param_dict — channel index of the [bs, 15, T]
# control_params tensor that the controller emits at sigmoid output.
_PEQ_BANDS = [
    {"name": "low_shelf",  "kind": "low_shelf", "ch_gain": 0,  "ch_freq": 1,  "ch_q": 2,
     "range_keys": ("low_shelf_gain_db",  "low_shelf_cutoff_freq",  "low_shelf_q_factor")},
    {"name": "band0",      "kind": "peaking",   "ch_gain": 3,  "ch_freq": 4,  "ch_q": 5,
     "range_keys": ("band0_gain_db",      "band0_cutoff_freq",      "band0_q_factor")},
    {"name": "band1",      "kind": "peaking",   "ch_gain": 6,  "ch_freq": 7,  "ch_q": 8,
     "range_keys": ("band1_gain_db",      "band1_cutoff_freq",      "band1_q_factor")},
    {"name": "band2",      "kind": "peaking",   "ch_gain": 9,  "ch_freq": 10, "ch_q": 11,
     "range_keys": ("band2_gain_db",      "band2_cutoff_freq",      "band2_q_factor")},
    {"name": "high_shelf", "kind": "high_shelf", "ch_gain": 12, "ch_freq": 13, "ch_q": 14,
     "range_keys": ("high_shelf_gain_db", "high_shelf_cutoff_freq", "high_shelf_q_factor")},
]


def _extract_parametric_eq(processor: torch.nn.Module, qname: str) -> DspBlockSpec:
    """Serialize a 5-band ParametricEQ's denorm ranges and channel layout.

    The controller emits a sigmoid in [0, 1] for each of the 15 control values;
    the C++ side denormalizes via per-band (gain, freq, Q) ranges to feed the
    biquad cascade. With ``freeze_freqs=True`` the freq ranges collapse to a
    single midpoint value — emitted as ``cutoff_freq`` for clarity.
    """
    if not getattr(processor, "freeze_freqs", False):
        raise ExportValidationError(
            f"{qname}: split-export only supports ParametricEQ(freeze_freqs=True). "
            "Per-band cutoff_freq must be a fixed scalar so the C++ side can "
            "precompute coefficients deterministically."
        )
    ranges = processor.param_ranges
    bands = []
    for spec in _PEQ_BANDS:
        gk, fk, qk = spec["range_keys"]
        f_lo, f_hi = ranges[fk]
        if f_lo != f_hi:
            raise ExportValidationError(
                f"{qname}: band {spec['name']}: freeze_freqs=True but freq range "
                f"is non-degenerate [{f_lo}, {f_hi}]."
            )
        bands.append({
            "name": spec["name"],
            "kind": spec["kind"],
            "cutoff_freq": float(f_lo),
            "gain_db_range": [float(ranges[gk][0]), float(ranges[gk][1])],
            "q_range": [float(ranges[qk][0]), float(ranges[qk][1])],
            "param_channels": {
                "gain": spec["ch_gain"],
                "q": spec["ch_q"],
            },
        })
    return DspBlockSpec(
        kind="parametric_eq_5band",
        name=qname,
        params={
            "sample_rate": int(processor.sample_rate),
            "block_size": int(processor.block_size),
            "num_control_params": int(processor.num_control_params),
            "bands": bands,
        },
    )


_EXTRACTORS = {
    "StaticRationalNonlinearity": _extract_rational_a,
    "ParametricEQ": _extract_parametric_eq,
}


def _has_learned_controller(model: GreyBoxModel) -> bool:
    """Any non-Dummy controller in the chain means we need an ONNX trace."""
    return any(not isinstance(c, DummyController) for c in model.controller.controllers)


# ---------------------------------------------------------------------------
# nn+dsp ONNX trace path: controller-only graph
# ---------------------------------------------------------------------------


class _ControllerWrapper(torch.nn.Module):
    """Trace wrapper for a single non-Dummy controller. Inputs: audio block +
    flat LSTM (h, c) state pairs (one per stateful submodule found inside the
    controller). Outputs: ``[1, num_control_params, T]`` params + new states."""

    def __init__(self, controller: torch.nn.Module, entries: List[StatefulEntry]):
        super().__init__()
        self.controller = controller
        self._entries = entries

    def forward(self, audio_in, *states):
        seed_states(self._entries, states)
        params = self.controller(audio_in)
        return (params, *collect_new_states(self._entries))


def _export_controller_onnx(
    controller: torch.nn.Module,
    proc_index: int,
    out_path: Path,
    block_len: int,
) -> tuple[List[StatefulEntry], List[str], List[str]]:
    """Trace a single controller to ONNX at a FIXED time length.

    Deliberately fully-static shape: torch.onnx (TorchScript path) is buggy
    with ``dynamic_axes`` on the time dim of an LSTM — it produces graphs that
    appear correct but emit garbage on most channels (we verified this; max
    abs err ~0.28 on sigmoid output vs ~6e-8 fully-static). The C++ host calls
    the controller once per block at its natural ``block_size`` rate, so a
    fixed time axis matches the streaming contract anyway.

    Returns (entries, in_names, out_names) so the meta can advertise state IO.
    """
    entries = collect_stateful(controller)
    if not entries:
        raise ExportValidationError(
            f"controller for processor {proc_index} has no stateful (LSTM) "
            "submodules; either it isn't a learned controller or this exporter "
            "doesn't yet recognize its inner LSTM."
        )
    wrapper = _ControllerWrapper(controller, entries).eval()

    # Use a deterministic non-zero pattern (literal, not RNG-derived). With
    # all-zero audio + zero states, do_constant_folding collapses parts of
    # the audio branch and the resulting graph diverges from PyTorch on real
    # inputs. RNG-based audio is correct but introduces process-state
    # dependence into the trace, which is hard to debug.
    audio = torch.linspace(-0.1, 0.1, block_len, dtype=torch.float32).view(1, 1, block_len)
    states = [
        torch.zeros(e.num_layers, 1, e.hidden_size)
        for e in entries
        for _ in (0, 1)
    ]
    example = (audio, *states)

    in_names = ["audio_in"]
    for e in entries:
        in_names.append(f"{e.onnx_base}_h_in")
        in_names.append(f"{e.onnx_base}_c_in")
    params_name = f"params_proc_{proc_index}"
    out_names = [params_name]
    for e in entries:
        out_names.append(f"{e.onnx_base}_h_out")
        out_names.append(f"{e.onnx_base}_c_out")

    # reset hidden_state flag so trace doesn't bake in init=True
    for e in entries:
        if hasattr(e.module, "is_hidden_state_init"):
            e.module.is_hidden_state_init = False

    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            example,
            str(out_path),
            opset_version=17,
            do_constant_folding=True,
            input_names=in_names,
            output_names=out_names,
            # No dynamic_axes — see docstring.
            dynamo=False,
        )
    return entries, in_names, out_names


def export_grey_bundle(inputs: ExportInputs) -> PluginMeta:
    """Serialize a trained ``GreyBoxModel`` into a staging bundle.

    Bundle layout:
      - ``plugin_meta.json`` with ``stage_kind`` reflecting NN-vs-DSP content.
      - ``source.hydra.yaml`` (the training config, verbatim).
      - ``model.onnx`` if any controller in the chain is non-Dummy.
    """
    run_dir = inputs.run_dir.resolve()
    out_dir = inputs.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    hydra_config = run_dir / ".hydra" / "config.yaml"
    cfg: DictConfig = OmegaConf.load(hydra_config)  # type: ignore[assignment]

    system = _load_system_and_weights(run_dir, ckpt_path=inputs.ckpt_path)
    model = system.model
    if not isinstance(model, GreyBoxModel):
        raise ExportValidationError(
            f"export_grey_bundle requires GreyBoxModel; got {type(model).__name__}"
        )
    if int(model.num_controls) != 0:
        raise ExportValidationError(
            "split-export currently only supports chains with num_controls=0 "
            f"(no host-exposed knobs on the model itself); got {model.num_controls}."
        )

    # Match BB export: drop any carried-over controller/LSTM state so the
    # trace is taken from a clean init. Without this, residual hidden_state
    # from training-time mutations bleeds into the traced graph and the
    # exported ONNX produces different outputs from a fresh-init forward.
    model.reset_states()

    processors = list(model.processor.processors)
    validate_grey_exportable(processors)

    dsp_blocks: list[DspBlockSpec] = []
    for i, proc in enumerate(processors):
        extract = _EXTRACTORS[type(proc).__name__]
        dsp_blocks.append(extract(proc, qname=f"processor.processors.{i}"))

    sample_rate = int(cfg.data.sample_rate)
    model_id = _model_id_from_run(run_dir, cfg)
    effect_name = inputs.effect_name or (
        inputs.effect_cfg.get("name") if inputs.effect_cfg else model_id
    )

    state_tensors: list[StateSpec] = []
    in_names: list[str] = []
    out_names: list[str] = []
    architecture = "dsp"
    stage_kind = "dsp"

    if _has_learned_controller(model):
        # Currently only handle the single-controller case (auto-EQ shape).
        # Multi-controller chains would emit one ONNX per controller or
        # concat outputs into a packed tensor; defer until we have that case.
        nondummy = [
            (i, c) for i, c in enumerate(model.controller.controllers)
            if not isinstance(c, DummyController)
        ]
        if len(nondummy) != 1:
            raise ExportValidationError(
                f"split-export of nn+dsp chains supports exactly one non-Dummy "
                f"controller for now; got {len(nondummy)}."
            )
        proc_idx, controller = nondummy[0]

        # Trace at exactly the controller's natural block_size: one LSTM
        # step per ONNX call, matching the streaming contract on the C++ side.
        # Larger trace lengths would just pre-multiply work and bake in a
        # static nsteps that the host couldn't vary anyway (no dynamic axes).
        block_size = int(getattr(controller, "block_size", 128))
        entries, in_names, out_names = _export_controller_onnx(
            controller, proc_idx, out_dir / "model.onnx", block_size
        )
        state_tensors = [
            StateSpec(
                name=f"{e.onnx_base}_{hc}",
                shape=[e.num_layers, 1, e.hidden_size],
                dtype="float32",
            )
            for e in entries
            for hc in ("h", "c")
        ]
        architecture = "lstm"
        stage_kind = "nn+dsp"

    meta = PluginMeta(
        effect_name=str(effect_name),
        model_id=model_id,
        architecture=architecture,
        sample_rate=sample_rate,
        channels=1,
        causal=True,
        receptive_field=1,
        latency_samples=latency_for(True, 1),
        num_controls=0,
        stage_kind=stage_kind,
        state_tensors=state_tensors,
        input_names=in_names,
        output_names=out_names,
        dsp_blocks=dsp_blocks,
    )
    meta.write(out_dir / "plugin_meta.json")
    shutil.copyfile(hydra_config, out_dir / "source.hydra.yaml")
    return meta

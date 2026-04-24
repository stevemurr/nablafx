"""End-to-end export: Lightning run directory -> staging bundle.

The staging bundle is a directory containing:
  - ``model.onnx``          : the traced graph
  - ``plugin_meta.json``    : architecture, controls, state shapes
  - ``source.hydra.yaml``   : the training Hydra config, verbatim

This bundle is portable; the macOS CLAP build step consumes it to produce a
signed ``.clap`` plugin.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence

import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from nablafx.core.models import BlackBoxModel
from .meta import (
    ControlSpec,
    PluginMeta,
    StateSpec,
    controls_from_effect_yaml,
    latency_for,
)
from .validate import ExportValidationError, validate_exportable
from .wrapper import StatefulEntry, build_wrapper


# Arbitrary — the ONNX time axis is dynamic so the concrete length only has to
# satisfy the convolution kernel at every dilation level. The effective trace
# length used below is ``max(MIN_TRACE_BLOCK_LEN, rf + 512)`` so that even a
# deeply dilated TCN has enough samples to convolve against once we skip the
# internal zero-pad.
MIN_TRACE_BLOCK_LEN = 2048


@dataclass
class ExportInputs:
    """Concrete arguments to :func:`export_bundle`."""
    run_dir: Path
    out_dir: Path
    effect_name: Optional[str] = None      # human-readable; default = model_id
    effect_cfg: Optional[dict] = None      # parsed conf/effect/<name>.yaml (or None)
    letters_in_use: Optional[Sequence[str]] = None  # from data config / filenames


def _load_system_and_weights(run_dir: Path) -> torch.nn.Module:
    """Reconstruct the Lightning system from .hydra/config.yaml and load weights."""
    hydra_config = run_dir / ".hydra" / "config.yaml"
    ckpt_path = run_dir / "checkpoints" / "last.ckpt"
    if not hydra_config.is_file():
        raise FileNotFoundError(f"Missing {hydra_config}; run_dir must be a Hydra output dir")
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Missing {ckpt_path}")

    cfg: DictConfig = OmegaConf.load(hydra_config)  # type: ignore[assignment]
    system = instantiate(cfg.model, _convert_="all")
    # Lightning ckpt contains optimizer state, hparams, etc.; weights_only=False
    # is appropriate because the ckpt comes from our own trusted training runs.
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    system.load_state_dict(ckpt["state_dict"])
    system.eval()
    return system


def _architecture_tag(processor: torch.nn.Module) -> str:
    """Map the processor class to the short tag written into plugin_meta.json."""
    name = type(processor).__name__.lower()
    # Known names in the repo: TCN, LSTM, GCN, S4, ...
    for tag in ("tcn", "lstm", "gcn"):
        if name == tag:
            return tag
    raise ExportValidationError(
        f"Unsupported processor {type(processor).__name__}; v1 supports TCN, LSTM, GCN"
    )


def _receptive_field(processor: torch.nn.Module) -> int:
    """Samples of lookback the plugin's ring buffer must carry."""
    rf = getattr(processor, "rf", None)
    if rf is not None:
        return int(rf)
    # LSTM is strictly causal with no convolutional lookback; the only "state"
    # is its recurrent hidden state, carried across blocks explicitly.
    return 1


def _causal(processor: torch.nn.Module) -> bool:
    """TCN/GCN have a boolean; LSTM is causal by construction."""
    return bool(getattr(processor, "causal", True))


def _example_inputs(
    wrapper: torch.nn.Module,
    entries: List[StatefulEntry],
    num_controls: int,
    block_len: int,
) -> tuple:
    """Build positional example args matching wrapper.forward's signature."""
    audio = torch.zeros(1, 1, block_len)
    states = [torch.zeros(e.num_layers, 1, e.hidden_size) for e in entries for _ in (0, 1)]
    if num_controls > 0 and entries:
        return (audio, torch.zeros(1, num_controls), *states)
    if num_controls > 0:
        return (audio, torch.zeros(1, num_controls))
    if entries:
        return (audio, *states)
    return (audio,)


def _io_names(entries: List[StatefulEntry], num_controls: int) -> tuple[list[str], list[str]]:
    in_names = ["audio_in"]
    if num_controls > 0:
        in_names.append("controls")
    for e in entries:
        in_names.append(f"{e.onnx_base}_h_in")
        in_names.append(f"{e.onnx_base}_c_in")

    out_names = ["audio_out"]
    for e in entries:
        out_names.append(f"{e.onnx_base}_h_out")
        out_names.append(f"{e.onnx_base}_c_out")
    return in_names, out_names


def _dynamic_axes(in_names: list[str], out_names: list[str]) -> dict:
    """Only the time dim of the audio tensors is dynamic. State tensors keep
    their exact shape across calls; controls are fixed-shape per block."""
    axes: dict[str, dict[int, str]] = {
        "audio_in": {0: "batch", 2: "time"},
        "audio_out": {0: "batch", 2: "time"},
    }
    # State and controls remain statically shaped; don't advertise them.
    return axes


def _model_id_from_run(run_dir: Path, cfg: DictConfig) -> str:
    """A filesystem-safe id for the bundle: dataset + model stub + run stamp."""
    dataset = run_dir.parent.parent.name if run_dir.parent.parent.name else "unknown"
    # ``<artifact_root>/<dataset>/outputs/<date>/<time>/`` — two parents up gets
    # us the dataset slug. Fall back gracefully when the layout differs.
    stamp = f"{run_dir.parent.name}-{run_dir.name}"
    proc = cfg.model.model.processor._target_.rsplit(".", 1)[-1].lower()  # type: ignore[union-attr]
    return f"{dataset}_{proc}_{stamp}"


def export_bundle(inputs: ExportInputs) -> PluginMeta:
    """Trace the trained model to ONNX and write the staging bundle.

    Returns the :class:`PluginMeta` that was written, for callers that want to
    log or chain into the plugin-build step.
    """
    run_dir = inputs.run_dir.resolve()
    out_dir = inputs.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    hydra_config = run_dir / ".hydra" / "config.yaml"
    cfg: DictConfig = OmegaConf.load(hydra_config)  # type: ignore[assignment]

    system = _load_system_and_weights(run_dir)
    model: BlackBoxModel = system.model  # type: ignore[attr-defined]
    if not isinstance(model, BlackBoxModel):
        raise ExportValidationError(
            f"v1 only supports BlackBoxModel; got {type(model).__name__} "
            "(grey-box export is v2 work)"
        )

    validate_exportable(model)

    model.reset_states()
    wrapper, entries = build_wrapper(model)

    processor = model.processor
    arch = _architecture_tag(processor)
    rf = _receptive_field(processor)
    causal = _causal(processor)
    num_controls = int(model.num_controls)
    sample_rate = int(cfg.data.sample_rate)

    trace_len = max(MIN_TRACE_BLOCK_LEN, rf + 512)
    example = _example_inputs(wrapper, entries, num_controls, trace_len)
    in_names, out_names = _io_names(entries, num_controls)
    dyn_axes = _dynamic_axes(in_names, out_names)

    onnx_path = out_dir / "model.onnx"
    with torch.no_grad():
        # dynamo=False: legacy tracer; maps nn.LSTM cleanly to ONNX LSTM op.
        # opset 17 matches ONNX Runtime 1.16+ and covers every op we emit.
        torch.onnx.export(
            wrapper,
            example,
            str(onnx_path),
            opset_version=17,
            do_constant_folding=True,
            input_names=in_names,
            output_names=out_names,
            dynamic_axes=dyn_axes,
            dynamo=False,
        )

    # Build metadata
    controls: List[ControlSpec] = []
    if num_controls > 0:
        if inputs.effect_cfg is None or not inputs.letters_in_use:
            raise ExportValidationError(
                f"Model has num_controls={num_controls} but no effect yaml / "
                "letters_in_use were provided; cannot assign knob metadata."
            )
        controls = controls_from_effect_yaml(inputs.effect_cfg, inputs.letters_in_use)
        if len(controls) != num_controls:
            raise ExportValidationError(
                f"Effect yaml produced {len(controls)} controls but model expects "
                f"{num_controls}. Check params_idxs_to_use in the data config."
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

    model_id = _model_id_from_run(run_dir, cfg)
    effect_name = inputs.effect_name or (
        inputs.effect_cfg.get("name") if inputs.effect_cfg else model_id
    )
    meta = PluginMeta(
        effect_name=str(effect_name),
        model_id=model_id,
        architecture=arch,
        sample_rate=sample_rate,
        channels=1,
        causal=causal,
        receptive_field=rf,
        latency_samples=latency_for(causal, rf),
        num_controls=num_controls,
        controls=controls,
        state_tensors=state_tensors,
        input_names=in_names,
        output_names=out_names,
    )
    meta.write(out_dir / "plugin_meta.json")

    shutil.copyfile(hydra_config, out_dir / "source.hydra.yaml")
    return meta

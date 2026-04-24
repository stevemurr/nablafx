"""`plugin_meta.json` schema.

Written on the Python side, read on the C++ side by ``native/clap/src/meta.cpp``.
Keep the two in sync: bump ``schema_version`` on any breaking change and teach
the C++ loader about both versions.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import List, Sequence

SCHEMA_VERSION = 1


@dataclass(frozen=True)
class ControlSpec:
    id: str              # short id, e.g. "A"; letter from the dataset filename convention
    name: str            # human-readable, e.g. "Attack"
    min: float
    max: float
    default: float
    skew: float = 1.0    # 1.0 = linear; >1 = more resolution near min
    unit: str = ""       # free-form ("dB", "ms", "step", ...)


@dataclass(frozen=True)
class StateSpec:
    name: str            # ONNX input/output name (also used as pair id: "<name>_in" / "<name>_out")
    shape: List[int]     # (num_layers, batch, hidden_size) etc. batch == 1 in meta;
                         # plugin allocates batch=channels.
    dtype: str           # "float32"


@dataclass
class PluginMeta:
    effect_name: str
    model_id: str
    architecture: str    # "tcn" | "lstm" | "gcn"
    sample_rate: int
    channels: int        # 1 for mono model; plugin may run batch=2 for stereo
    causal: bool
    receptive_field: int        # total RF in samples; plugin ring-buffers (rf - 1)
    latency_samples: int        # reported to host via clap_plugin_latency
    num_controls: int
    controls: List[ControlSpec] = field(default_factory=list)
    state_tensors: List[StateSpec] = field(default_factory=list)
    input_names: List[str] = field(default_factory=list)
    output_names: List[str] = field(default_factory=list)
    schema_version: int = SCHEMA_VERSION

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)

    def write(self, path: Path) -> None:
        path.write_text(self.to_json() + "\n")


def latency_for(causal: bool, receptive_field: int) -> int:
    """Samples of processing latency to report to the host.

    Causal nets add none; non-causal symmetric padding looks `(rf - 1) // 2`
    samples into the future.
    """
    if causal:
        return 0
    return (receptive_field - 1) // 2


def controls_from_effect_yaml(
    effect_cfg: dict,
    letters_in_use: Sequence[str],
) -> List[ControlSpec]:
    """Join an ``effect`` yaml with the ordered control letters the model
    expects. ``letters_in_use`` comes from the training data config (the
    ``params_idxs_to_use`` selection + the filename letter order); an empty
    list means the model is non-parametric.
    """
    controls_block = effect_cfg.get("controls", {}) or {}
    result: List[ControlSpec] = []
    for letter in letters_in_use:
        if letter not in controls_block:
            raise KeyError(
                f"Effect yaml is missing a spec for letter {letter!r}. "
                f"Known letters: {list(controls_block)}"
            )
        spec = controls_block[letter]
        result.append(
            ControlSpec(
                id=letter,
                name=str(spec["name"]),
                min=float(spec.get("min", 0.0)),
                max=float(spec.get("max", 1.0)),
                default=float(spec.get("default", 0.5)),
                skew=float(spec.get("skew", 1.0)),
                unit=str(spec.get("unit", "")),
            )
        )
    return result

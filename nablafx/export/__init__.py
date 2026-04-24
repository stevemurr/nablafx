"""Plugin export pipeline: Lightning checkpoint -> ONNX + metadata bundle.

The bundle produced by :func:`export_bundle` is architecture-agnostic: it
contains ``model.onnx``, ``plugin_meta.json``, and ``source.hydra.yaml``, and
is consumed by the C++ CLAP plugin template in ``native/clap``.
"""

from .bundle import export_bundle
from .meta import PluginMeta, ControlSpec, StateSpec

__all__ = ["export_bundle", "PluginMeta", "ControlSpec", "StateSpec"]

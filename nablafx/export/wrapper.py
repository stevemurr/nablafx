"""Wraps a trained :class:`nablafx.BlackBoxModel` so it can be ONNX-exported as
a stateless, streaming-friendly graph.

Two transformations happen at trace time:

1. The top-level processor's internal causal/non-causal zero-padding is turned
   off by setting ``processor._export_skip_pad = True``. The ONNX consumer is
   expected to prepend ``rf - 1`` samples of real history from its own ring
   buffer on every block (see ``native/clap``).

2. Stateful submodules (``LSTM``, ``TVFiLMCond``, ``TFiLM``, ``TinyTFiLM``)
   normally store their LSTM hidden state on ``self.hidden_state`` and read it
   back on the next forward. For export we re-seed those attributes from
   explicit wrapper inputs before each forward and return the new states as
   explicit outputs. The originals' ``forward`` is unchanged; tracing simply
   follows the "use provided state" branch it already has.

The wrapper.forward signature is specialized by the (parametric?, stateful?)
cross-product — four small subclasses, one is picked by :func:`build_wrapper`.
This keeps ONNX input/output lists clean instead of padding with dummy zeros.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

import torch

from nablafx.core.models import BlackBoxModel
from nablafx.processors.components import TFiLM, TinyTFiLM, TVFiLMCond
from nablafx.processors.lstm import LSTM


STATEFUL_CLASSES: Tuple[type, ...] = (LSTM, TVFiLMCond, TFiLM, TinyTFiLM)


@dataclass
class StatefulEntry:
    qname: str                  # fully qualified module name within the model
    module: torch.nn.Module     # reference, used to seed/read hidden_state
    num_layers: int             # from m.lstm
    hidden_size: int            # from m.lstm
    onnx_base: str              # sanitized name used as the ONNX tensor prefix


def _sanitize(qname: str) -> str:
    """Produce a legal ONNX tensor name stem from a module path."""
    return qname.replace(".", "_").replace("-", "_") or "root"


def collect_stateful(model: torch.nn.Module) -> List[StatefulEntry]:
    """Walk the model in named_modules() order and pick up every module that
    the wrapper must seed/read at export time. Order matters: it fixes the
    positional layout of state tensors in the ONNX input list."""
    entries: List[StatefulEntry] = []
    seen_onnx_bases: set[str] = set()
    for qname, m in model.named_modules():
        if not isinstance(m, STATEFUL_CLASSES):
            continue
        if not hasattr(m, "lstm") or not isinstance(m.lstm, torch.nn.LSTM):
            # defensive: if someone adds a new stateful class without an inner
            # nn.LSTM, catch it instead of silently producing wrong shapes
            raise RuntimeError(
                f"{qname} ({type(m).__name__}) has no inner nn.LSTM; cannot "
                "infer export state shape."
            )
        base = _sanitize(qname)
        # Disambiguate if two qnames sanitize to the same string
        candidate = base
        n = 1
        while candidate in seen_onnx_bases:
            n += 1
            candidate = f"{base}_{n}"
        seen_onnx_bases.add(candidate)
        entries.append(
            StatefulEntry(
                qname=qname,
                module=m,
                num_layers=m.lstm.num_layers,
                hidden_size=m.lstm.hidden_size,
                onnx_base=candidate,
            )
        )
    return entries


def seed_states(entries: Sequence[StatefulEntry], flat_states: Sequence[torch.Tensor]) -> None:
    """Assign `(h, c)` tuples from `flat_states` onto each stateful module so
    its existing forward picks them up via `self.hidden_state`."""
    assert len(flat_states) == 2 * len(entries), (
        f"expected {2 * len(entries)} state tensors, got {len(flat_states)}"
    )
    for i, e in enumerate(entries):
        h = flat_states[2 * i]
        c = flat_states[2 * i + 1]
        e.module.hidden_state = (h, c)
        # LSTM and TVFiLMCond gate on an is_hidden_state_init bool; TFiLM /
        # TinyTFiLM gate on `hidden_state is None`. Setting both covers both.
        if hasattr(e.module, "is_hidden_state_init"):
            e.module.is_hidden_state_init = True


def collect_new_states(entries: Sequence[StatefulEntry]) -> List[torch.Tensor]:
    """Read back `(h, c)` tuples the forward pass left on each stateful module."""
    out: List[torch.Tensor] = []
    for e in entries:
        h, c = e.module.hidden_state
        out.append(h)
        out.append(c)
    return out


# ---------------------------------------------------------------------------
# Wrapper variants. Each has a fixed positional signature so torch.onnx.export
# can assign clean named inputs without trailing dummies.
# ---------------------------------------------------------------------------


class _BaseWrapper(torch.nn.Module):
    def __init__(self, model: BlackBoxModel, entries: List[StatefulEntry]):
        super().__init__()
        self.model = model
        # Not an nn.Module field — we don't want stateful entries to move with
        # .to() twice or end up in state_dict. They only hold references.
        self._entries = entries


class NonParamStatelessWrapper(_BaseWrapper):
    def forward(self, audio_in: torch.Tensor) -> torch.Tensor:
        # BlackBoxModel.forward((x, controls); num_controls==0 ignores controls.
        return self.model(audio_in, torch.empty(0))


class NonParamStatefulWrapper(_BaseWrapper):
    def forward(self, audio_in: torch.Tensor, *states: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        seed_states(self._entries, states)
        y = self.model(audio_in, torch.empty(0))
        return (y, *collect_new_states(self._entries))


class ParamStatelessWrapper(_BaseWrapper):
    def forward(self, audio_in: torch.Tensor, controls: torch.Tensor) -> torch.Tensor:
        return self.model(audio_in, controls)


class ParamStatefulWrapper(_BaseWrapper):
    def forward(
        self,
        audio_in: torch.Tensor,
        controls: torch.Tensor,
        *states: torch.Tensor,
    ) -> Tuple[torch.Tensor, ...]:
        seed_states(self._entries, states)
        y = self.model(audio_in, controls)
        return (y, *collect_new_states(self._entries))


def build_wrapper(model: BlackBoxModel) -> Tuple[_BaseWrapper, List[StatefulEntry]]:
    """Pick the right wrapper variant for this model and flip streaming flags
    on the top-level processor. The returned wrapper is in eval mode and
    ready for :func:`torch.onnx.export`."""
    processor = model.processor
    # The flag is read via getattr(..., False) inside TCN/GCN.forward so
    # processors that don't know about it are unaffected.
    processor._export_skip_pad = True  # type: ignore[attr-defined]

    entries = collect_stateful(model)
    num_controls = int(model.num_controls)

    if num_controls > 0 and entries:
        wrapper: _BaseWrapper = ParamStatefulWrapper(model, entries)
    elif num_controls > 0:
        wrapper = ParamStatelessWrapper(model, entries)
    elif entries:
        wrapper = NonParamStatefulWrapper(model, entries)
    else:
        wrapper = NonParamStatelessWrapper(model, entries)

    wrapper.eval()
    return wrapper, entries

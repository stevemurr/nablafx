"""Pre-export validation: reject modules that torch.onnx cannot trace or that
would produce graphs the streaming CLAP plugin cannot run block-by-block."""

from __future__ import annotations

import torch


# Class names (not types — rational-activations is optional and we don't want
# to import it here) that block a clean ONNX export.
_BLOCKED_CLASS_NAMES = {
    "Rational",  # rational-activations; not in ONNX op set
}

# Architecture flags that v1 cannot stream. These are top-level processors or
# grey-box elements; their presence means the whole model is rejected.
_UNSUPPORTED_TOP_LEVEL = {
    "S4",           # fftconv is full-sequence, not causal-streamable
    "GreyBoxModel", # FSM-based IIR approximations; v2 work
}


class ExportValidationError(ValueError):
    """Raised when a model cannot be cleanly exported to streaming ONNX."""


def validate_exportable(model: torch.nn.Module) -> None:
    """Walk ``model`` and raise ExportValidationError if anything blocks export.

    Reasons we reject early (before torch.onnx.export so the error is legible):
      - Rational activations (not in ONNX op set).
      - S4 / grey-box processors (use full-sequence or frequency-sampling ops
        that don't survive block-wise streaming; v2 work).
    """
    problems: list[str] = []

    for qname, m in model.named_modules():
        cls = type(m).__name__
        if cls in _BLOCKED_CLASS_NAMES:
            problems.append(
                f"  - {qname or '<root>'} is a {cls}; replace `act_type=rational` "
                "with `tanh` or `prelu` in the model config and retrain, "
                "or implement a custom ONNX op."
            )
        if cls in _UNSUPPORTED_TOP_LEVEL:
            problems.append(
                f"  - {qname or '<root>'} is a {cls}; this architecture is not "
                "supported by v1 of the CLAP export pipeline."
            )

    if problems:
        raise ExportValidationError(
            "Model contains modules that cannot be exported for streaming:\n"
            + "\n".join(problems)
        )

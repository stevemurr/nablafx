"""PyTorch vs ONNX Runtime parity tests for the export wrapper.

Each test constructs a small model, wraps it with
:func:`nablafx.export.wrapper.build_wrapper`, traces it with
``torch.onnx.export``, then compares the ONNX session output to the PyTorch
wrapper output on the same input + state. They must agree to ~1e-5 — the only
source of difference is the onnxruntime LSTM kernel's rounding.

Skipped automatically if ``onnx`` / ``onnxruntime`` aren't installed (the
export optional-dependency group).
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from nablafx.core.models import BlackBoxModel
from nablafx.export.wrapper import build_wrapper
from nablafx.processors.lstm import LSTM
from nablafx.processors.tcn import TCN
from nablafx.processors.gcn import GCN

ort = pytest.importorskip("onnxruntime")
_ = pytest.importorskip("onnx")


_BLOCK_LEN = 1024
_RTOL = 1e-4
_ATOL = 1e-5


def _tiny_tcn(num_controls: int = 0, causal: bool = True) -> TCN:
    return TCN(
        num_inputs=1,
        num_outputs=1,
        num_controls=num_controls,
        num_blocks=3,
        kernel_size=5,
        dilation_growth=2,
        channel_width=8,
        stack_size=10,
        causal=causal,
        cond_type=None if num_controls == 0 else "film",
        act_type="tanh",
    )


def _tiny_gcn(num_controls: int = 0, causal: bool = True) -> GCN:
    return GCN(
        num_inputs=1,
        num_outputs=1,
        num_controls=num_controls,
        num_blocks=3,
        kernel_size=5,
        dilation_growth=2,
        channel_width=8,
        stack_size=10,
        causal=causal,
        cond_type=None if num_controls == 0 else "film",
    )


def _tiny_lstm(num_controls: int = 0) -> LSTM:
    return LSTM(
        num_inputs=1,
        num_outputs=1,
        num_controls=num_controls,
        hidden_size=8,
        num_layers=1,
        cond_type=None if num_controls == 0 else "fixed",
    )


def _export_and_load(wrapper, example, in_names, out_names, tmp_path):
    onnx_path = tmp_path / "model.onnx"
    dyn_axes = {
        "audio_in": {0: "batch", 2: "time"},
        "audio_out": {0: "batch", 2: "time"},
    }
    with torch.no_grad():
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
    return ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])


def _assert_close(a: torch.Tensor, b: np.ndarray) -> None:
    np.testing.assert_allclose(a.detach().cpu().numpy(), b, rtol=_RTOL, atol=_ATOL)


def test_tcn_nonparam_causal_parity(tmp_path):
    torch.manual_seed(0)
    processor = _tiny_tcn(num_controls=0, causal=True)
    model = BlackBoxModel(processor)
    model.eval()

    wrapper, entries = build_wrapper(model)
    assert len(entries) == 0

    audio = torch.randn(1, 1, _BLOCK_LEN)
    example = (audio,)

    in_names = ["audio_in"]
    out_names = ["audio_out"]
    sess = _export_and_load(wrapper, example, in_names, out_names, tmp_path)

    with torch.no_grad():
        y_torch = wrapper(audio)

    y_ort = sess.run(["audio_out"], {"audio_in": audio.numpy()})[0]

    _assert_close(y_torch, y_ort)


def test_gcn_nonparam_causal_parity(tmp_path):
    torch.manual_seed(0)
    processor = _tiny_gcn(num_controls=0, causal=True)
    model = BlackBoxModel(processor)
    model.eval()

    wrapper, entries = build_wrapper(model)
    assert len(entries) == 0

    audio = torch.randn(1, 1, _BLOCK_LEN)
    sess = _export_and_load(wrapper, (audio,), ["audio_in"], ["audio_out"], tmp_path)

    with torch.no_grad():
        y_torch = wrapper(audio)
    y_ort = sess.run(["audio_out"], {"audio_in": audio.numpy()})[0]

    _assert_close(y_torch, y_ort)


def test_lstm_nonparam_parity(tmp_path):
    torch.manual_seed(0)
    processor = _tiny_lstm(num_controls=0)
    model = BlackBoxModel(processor)
    model.eval()

    wrapper, entries = build_wrapper(model)
    assert len(entries) == 1  # the LSTM itself

    audio = torch.randn(1, 1, _BLOCK_LEN)
    h0 = torch.randn(entries[0].num_layers, 1, entries[0].hidden_size)
    c0 = torch.randn(entries[0].num_layers, 1, entries[0].hidden_size)
    example = (audio, h0, c0)

    base = entries[0].onnx_base
    in_names = ["audio_in", f"{base}_h_in", f"{base}_c_in"]
    out_names = ["audio_out", f"{base}_h_out", f"{base}_c_out"]
    sess = _export_and_load(wrapper, example, in_names, out_names, tmp_path)

    with torch.no_grad():
        y_torch, h_torch, c_torch = wrapper(audio, h0, c0)

    outs = sess.run(
        out_names,
        {
            "audio_in": audio.numpy(),
            f"{base}_h_in": h0.numpy(),
            f"{base}_c_in": c0.numpy(),
        },
    )

    _assert_close(y_torch, outs[0])
    _assert_close(h_torch, outs[1])
    _assert_close(c_torch, outs[2])


def test_lstm_fixed_param_parity(tmp_path):
    torch.manual_seed(0)
    num_controls = 2
    processor = _tiny_lstm(num_controls=num_controls)
    model = BlackBoxModel(processor)
    model.eval()

    wrapper, entries = build_wrapper(model)
    assert len(entries) == 1

    audio = torch.randn(1, 1, _BLOCK_LEN)
    ctrl = torch.rand(1, num_controls)
    h0 = torch.randn(entries[0].num_layers, 1, entries[0].hidden_size)
    c0 = torch.randn(entries[0].num_layers, 1, entries[0].hidden_size)
    example = (audio, ctrl, h0, c0)

    base = entries[0].onnx_base
    in_names = ["audio_in", "controls", f"{base}_h_in", f"{base}_c_in"]
    out_names = ["audio_out", f"{base}_h_out", f"{base}_c_out"]
    sess = _export_and_load(wrapper, example, in_names, out_names, tmp_path)

    with torch.no_grad():
        y_torch, h_torch, c_torch = wrapper(audio, ctrl, h0, c0)

    outs = sess.run(
        out_names,
        {
            "audio_in": audio.numpy(),
            "controls": ctrl.numpy(),
            f"{base}_h_in": h0.numpy(),
            f"{base}_c_in": c0.numpy(),
        },
    )

    _assert_close(y_torch, outs[0])
    _assert_close(h_torch, outs[1])
    _assert_close(c_torch, outs[2])


@pytest.mark.parametrize("cond_type", ["film", "tfilm", "tvfilm"])
def test_tcn_param_parity(tmp_path, cond_type):
    """Parametric TCN variants (film is stateless MLP conditioning; tfilm and
    tvfilm pull in stateful TFiLM/TVFiLMCond LSTMs) must all export with
    PyTorch↔ORT parity."""
    torch.manual_seed(0)
    num_controls = 2
    processor = TCN(
        num_inputs=1,
        num_outputs=1,
        num_controls=num_controls,
        num_blocks=3,
        kernel_size=5,
        dilation_growth=2,
        channel_width=8,
        stack_size=10,
        causal=True,
        cond_type=cond_type,
        cond_block_size=32,
        cond_num_layers=1,
        act_type="tanh",
    )
    model = BlackBoxModel(processor)
    model.eval()

    wrapper, entries = build_wrapper(model)

    # film is stateless; tfilm adds one TFiLM per block; tvfilm adds one
    # TVFiLMCond at the top plus a TVFiLMMod per block.
    audio = torch.randn(1, 1, _BLOCK_LEN)
    ctrl = torch.rand(1, num_controls)
    states: list[torch.Tensor] = []
    in_names = ["audio_in", "controls"]
    out_names = ["audio_out"]
    for e in entries:
        h = torch.randn(e.num_layers, 1, e.hidden_size)
        c = torch.randn(e.num_layers, 1, e.hidden_size)
        states.extend([h, c])
        in_names.extend([f"{e.onnx_base}_h_in", f"{e.onnx_base}_c_in"])
        out_names.extend([f"{e.onnx_base}_h_out", f"{e.onnx_base}_c_out"])

    example = (audio, ctrl, *states)
    sess = _export_and_load(wrapper, example, in_names, out_names, tmp_path)

    with torch.no_grad():
        outputs = wrapper(audio, ctrl, *states)
    y_torch = outputs[0] if isinstance(outputs, tuple) else outputs

    feeds = {"audio_in": audio.numpy(), "controls": ctrl.numpy()}
    for i, e in enumerate(entries):
        feeds[f"{e.onnx_base}_h_in"] = states[2 * i].numpy()
        feeds[f"{e.onnx_base}_c_in"] = states[2 * i + 1].numpy()

    outs = sess.run(out_names, feeds)
    _assert_close(y_torch, outs[0])


def test_tcn_streaming_matches_one_shot(tmp_path):
    """Feeding a long signal in two blocks with ``rf-1`` samples of lookback
    overlap must reproduce the one-shot output exactly — this is the contract
    the CLAP plugin's ring buffer relies on."""
    torch.manual_seed(0)
    processor = _tiny_tcn(num_controls=0, causal=True)
    model = BlackBoxModel(processor)
    model.eval()

    wrapper, _ = build_wrapper(model)
    rf = processor.rf
    block = 512

    audio = torch.randn(1, 1, rf - 1 + block * 2)
    example = (audio[:, :, : rf - 1 + block],)  # shape used for trace only
    sess = _export_and_load(wrapper, example, ["audio_in"], ["audio_out"], tmp_path)

    a = audio.numpy()
    out1 = sess.run(None, {"audio_in": a[:, :, : rf - 1 + block]})[0]
    out2 = sess.run(None, {"audio_in": a[:, :, block : rf - 1 + block * 2]})[0]
    one = sess.run(None, {"audio_in": a})[0]

    assert out1.shape[-1] == block
    np.testing.assert_array_equal(out1, one[:, :, :block])
    np.testing.assert_array_equal(out2, one[:, :, block : block * 2])


def test_tcn_rejects_rational_activation():
    """Rational activations aren't in the ONNX op set; the validator rejects
    them before we ever touch torch.onnx.export."""
    from nablafx.export.validate import ExportValidationError, validate_exportable

    class FakeRational(torch.nn.Module):
        def forward(self, x):  # pragma: no cover (trivial stub)
            return x

    FakeRational.__name__ = "Rational"

    class Wrap(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.act = FakeRational()

    with pytest.raises(ExportValidationError, match="rational"):
        validate_exportable(Wrap())

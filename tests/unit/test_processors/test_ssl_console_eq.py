"""Unit tests for the SSL 9000 J console EQ grey-box processor.

Locks the behaviours verified during Phase-2 bring-up: correct control layout,
neutral pass-through, band placement/magnitude, shelf<->bell blend, and
differentiability. The biquad math mirrors nablafx.processors.dsp.biquad (and
the C++ runtime), so a pure-bell magnitude equals the closed-form RBJ response.
"""
import math

import numpy as np
import torch

from nablafx.processors import SSLConsoleEQ

SR = 48000.0


def _eq():
    return SSLConsoleEQ(sample_rate=SR, control_type="static-cond")


def _norm(eq, key, val):
    lo, hi = eq.param_ranges[key]
    return 0.0 if hi == lo else (val - lo) / (hi - lo)


def _make(eq, **phys):
    d = dict(hpf_freq=10.0, hpf_q=0.7, lf_gain=0.0, lf_freq=100.0, lf_q=0.7, lf_bellmix=0.0,
             lmf_gain=0.0, lmf_freq=1000.0, lmf_q=1.0, hmf_gain=0.0, hmf_freq=3000.0, hmf_q=1.0,
             hf_gain=0.0, hf_freq=10000.0, hf_q=0.7, hf_bellmix=0.0, lpf_freq=23000.0, lpf_q=0.7)
    d.update(phys)
    v = torch.zeros(1, eq.num_control_params, 1)
    for i, k in enumerate(eq._keys):
        v[0, i, 0] = d[k] if k in ("lf_bellmix", "hf_bellmix") else _norm(eq, k, d[k])
    return v


def test_control_layout():
    eq = _eq()
    assert eq.num_control_params == 18
    assert eq._keys[0] == "hpf_freq" and eq._keys[-1] == "lpf_q"


def test_forward_shapes_and_finite():
    eq = _eq()
    x = torch.randn(2, 1, 4800) * 0.1
    y, pd = eq(x, _make(eq).repeat(2, 1, 1))
    assert y.shape == (2, 1, 4800)
    assert torch.isfinite(y).all()
    assert {"lf_gain", "lmf_gain", "lf_bellmix"} <= set(pd)


def test_neutral_is_flat_midband():
    eq = _eq()
    f = torch.tensor(np.geomspace(100, 10000, 200), dtype=torch.float32)
    mdb = eq.magnitude_db(_make(eq), f)[0].abs()
    assert mdb.max().item() < 0.1


def test_bell_placement_and_gain():
    eq = _eq()
    f = torch.tensor(np.geomspace(20, 20000, 400), dtype=torch.float32)
    mdb = eq.magnitude_db(_make(eq, lmf_gain=12.0, lmf_freq=1000.0, lmf_q=1.5), f)[0]
    assert abs(mdb.max().item() - 12.0) < 0.3
    assert abs(f[mdb.argmax()].item() - 1000.0) / 1000.0 < 0.05


def test_shelf_vs_bell_blend():
    eq = _eq()
    f = torch.tensor([20.0], dtype=torch.float32)
    shelf = eq.magnitude_db(_make(eq, lf_gain=10.0, lf_freq=120.0, lf_bellmix=0.0), f)[0, 0]
    bell = eq.magnitude_db(_make(eq, lf_gain=10.0, lf_freq=120.0, lf_bellmix=1.0), f)[0, 0]
    # a low shelf holds gain below its corner; a bell rolls back to ~0 dB there
    assert shelf.item() > 6.0 and bell.item() < 2.0


def test_matches_closed_form_rbj_peaking():
    eq = _eq()
    f0, g, q = 800.0, 8.0, 1.2
    f = torch.tensor(np.geomspace(50, 18000, 300), dtype=torch.float32)
    got = eq.magnitude_db(_make(eq, lmf_gain=g, lmf_freq=f0, lmf_q=q), f)[0].numpy()
    # closed-form RBJ peaking magnitude
    A = 10 ** (g / 40.0)
    w0 = 2 * math.pi * f0 / SR
    alpha = math.sin(w0) / (2 * q)
    cw = math.cos(w0)
    b = np.array([1 + alpha * A, -2 * cw, 1 - alpha * A])
    a = np.array([1 + alpha / A, -2 * cw, 1 - alpha / A])
    w = 2 * math.pi * f.numpy() / SR
    z = np.exp(-1j * w)
    H = (b[0] + b[1] * z + b[2] * z * z) / (a[0] + a[1] * z + a[2] * z * z)
    ref = 20 * np.log10(np.abs(H))
    assert np.max(np.abs(got - ref)) < 1e-2


def test_gradients_flow():
    eq = _eq()
    cp = _make(eq, lmf_gain=6.0).clone().requires_grad_(True)
    eq(torch.randn(1, 1, 2048) * 0.1, cp, train=True)[0].pow(2).mean().backward()
    assert torch.isfinite(cp.grad).all()
    assert cp.grad.abs().sum() > 0

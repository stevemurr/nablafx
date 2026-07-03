"""Tests for the SSL EQ transfer-function magnitude loss and its coupling to
SSLConsoleEQ.magnitude_db (the Phase-3 supervision signal)."""
import numpy as np
import torch

from nablafx.evaluation.tf_loss import TransferFunctionMagLoss, predicted_tf_db
from nablafx.processors import SSLConsoleEQ

# eq-dataset TF grid: log 10 Hz -> 24 kHz, 256 pts (see prepare_ssl_eq_data.py).
GRID = 10.0 * (2400.0) ** (np.arange(256) / 255.0)


def test_zero_when_equal():
    loss = TransferFunctionMagLoss(GRID)
    a = torch.randn(4, 256)
    assert loss(a, a.clone()).item() == 0.0


def test_l1_value_and_band_mask():
    loss = TransferFunctionMagLoss(GRID, band=(20.0, 20000.0), p=1)
    pred = torch.zeros(1, 256)
    target = torch.full((1, 256), 2.0)          # uniform 2 dB error
    # in-band mean |Δ| == 2.0; out-of-band bins ignored
    assert abs(loss(pred, target).item() - 2.0) < 1e-5
    # a large error placed only below 20 Hz must be ignored
    target2 = torch.zeros(1, 256)
    target2[0, GRID < 20.0] = 100.0
    assert loss(pred, target2).item() < 1e-6


def test_l2_mode():
    loss = TransferFunctionMagLoss(GRID, p=2)
    pred = torch.zeros(1, 256)
    target = torch.full((1, 256), 3.0)
    assert abs(loss(pred, target).item() - 9.0) < 1e-4   # mean Δ² = 9


def test_gradients_flow():
    loss = TransferFunctionMagLoss(GRID)
    pred = torch.zeros(1, 256, requires_grad=True)
    target = torch.full((1, 256), 1.5)
    loss(pred, target).backward()
    assert pred.grad is not None and pred.grad.abs().sum() > 0


def test_integration_with_processor_drops_toward_target():
    """A gradient step on the controller params should reduce the TF loss against
    the measured curve of a target SSL setting — proving magnitude_db is a usable
    differentiable supervision path."""
    eq = SSLConsoleEQ(sample_rate=48000.0, control_type="static-cond")
    f = torch.tensor(GRID, dtype=torch.float32)
    loss_fn = TransferFunctionMagLoss(GRID)

    def cp(**phys):
        d = dict(hpf_freq=10.0, hpf_q=0.7, lf_gain=0.0, lf_freq=100.0, lf_q=0.7, lf_bellmix=0.0,
                 lmf_gain=0.0, lmf_freq=1000.0, lmf_q=1.0, hmf_gain=0.0, hmf_freq=3000.0, hmf_q=1.0,
                 hf_gain=0.0, hf_freq=10000.0, hf_q=0.7, hf_bellmix=0.0, lpf_freq=23000.0, lpf_q=0.7)
        d.update(phys)
        v = torch.zeros(1, eq.num_control_params, 1)
        for i, k in enumerate(eq._keys):
            lo, hi = eq.param_ranges[k]
            v[0, i, 0] = d[k] if k in ("lf_bellmix", "hf_bellmix") else (0.0 if hi == lo else (d[k] - lo) / (hi - lo))
        return v

    target_db = predicted_tf_db(eq, cp(lmf_gain=8.0, lmf_freq=800.0, lmf_q=1.2), f).detach()
    p = cp().clone().requires_grad_(True)             # start flat
    l0 = loss_fn(predicted_tf_db(eq, p, f), target_db)
    opt = torch.optim.Adam([p], lr=0.05)
    for _ in range(150):
        opt.zero_grad()
        l = loss_fn(predicted_tf_db(eq, p, f), target_db)
        l.backward()
        opt.step()
    l1 = loss_fn(predicted_tf_db(eq, p, f), target_db)
    assert l1.item() < 0.25 * l0.item()               # loss collapses toward 0

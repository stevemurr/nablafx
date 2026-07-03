"""Transfer-function magnitude loss for the SSL console EQ grey-box.

The eq-dataset ships a measured 256-point magnitude curve (`sweep_qa.tf_mag_db`)
per example. Because an EQ is linear, that curve is a direct, content-free
supervision signal for the model's magnitude response — far cleaner and cheaper
than program-audio spectral losses. This module compares the model's predicted
magnitude (from ``SSLConsoleEQ.magnitude_db``) against the measured curve, in dB,
over the audible band.

Wiring (grey-box training step): with ``controls`` the knob vector and ``x`` a
block of audio,
    control_params = model.controller(x, controls)
    pred_db        = model.processor.processors[0].magnitude_db(control_params, freqs)
    loss_tf        = tf_loss(pred_db, target_tf_db)          # target from the sidecar
The measured ``target_tf_db`` per example is available from
``SSLParametricPluginDataset.tf_by_name`` (thread it into the batch to use this).
"""
from __future__ import annotations

from typing import Tuple

import torch


class TransferFunctionMagLoss(torch.nn.Module):
    """dB-domain magnitude error between predicted and measured transfer curves.

    Args:
        freqs_hz: (n_freq,) frequencies the two curves are sampled at.
        band:     (lo, hi) Hz — score only this band (the grid extremes are
                  degenerate: Nyquist has a bilinear-lowpass zero, sub-20 Hz is
                  unreliable). Matches the Phase-0 analytic scoring band.
        p:        1 -> mean |Δ| (L1, robust);  2 -> mean Δ² (L2).
    """

    def __init__(self, freqs_hz, band: Tuple[float, float] = (20.0, 20000.0), p: int = 1):
        super().__init__()
        f = torch.as_tensor(freqs_hz, dtype=torch.float32).flatten()
        mask = (f >= band[0]) & (f <= band[1])
        if not bool(mask.any()):
            raise ValueError(f"no grid points in band {band}")
        self.register_buffer("freqs", f)
        self.register_buffer("mask", mask.float())
        self.p = int(p)

    def forward(self, pred_db: torch.Tensor, target_db: torch.Tensor) -> torch.Tensor:
        if pred_db.shape[-1] != self.freqs.shape[0]:
            raise ValueError(f"pred has {pred_db.shape[-1]} freqs, expected {self.freqs.shape[0]}")
        diff = pred_db - target_db
        w = self.mask.to(diff.dtype)
        num = (diff.abs() if self.p == 1 else diff.pow(2)) * w
        return num.sum(dim=-1).div(w.sum()).mean()


def predicted_tf_db(eq_processor, control_params: torch.Tensor, freqs_hz) -> torch.Tensor:
    """Convenience: predicted (bs, n_freq) dB magnitude from an SSLConsoleEQ-like
    processor exposing ``magnitude_db(control_params, freqs_hz)``. ``control_params``
    is the controller output (bs, num_control_params, 1)."""
    f = torch.as_tensor(freqs_hz, dtype=control_params.dtype, device=control_params.device)
    return eq_processor.magnitude_db(control_params, f)

"""
Time Domain Evaluation Functions

This module registers time-domain evaluation functions from PyTorch and auraloss
that can be used as both losses and metrics.
"""

import torch
import auraloss
import torchmetrics as tm
from ..registry import register_function


# PyTorch built-in loss functions
@register_function("l1_loss", differentiable=True)
class L1Loss(torch.nn.Module):
    """L1 (Mean Absolute Error) loss function wrapper."""

    def __init__(self, reduction: str = "mean"):
        super().__init__()
        self.loss_fn = torch.nn.L1Loss(reduction=reduction)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.loss_fn(pred, target)


@register_function("mse_loss", differentiable=True)
class MSELoss(torch.nn.Module):
    """Mean Squared Error loss function wrapper."""

    def __init__(self, reduction: str = "mean"):
        super().__init__()
        self.loss_fn = torch.nn.MSELoss(reduction=reduction)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.loss_fn(pred, target)


@register_function("smooth_l1_loss", differentiable=True)
class SmoothL1Loss(torch.nn.Module):
    """Smooth L1 loss function wrapper."""

    def __init__(self, reduction: str = "mean", beta: float = 1.0):
        super().__init__()
        self.loss_fn = torch.nn.SmoothL1Loss(reduction=reduction, beta=beta)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.loss_fn(pred, target)


@register_function("huber_loss", differentiable=True)
class HuberLoss(torch.nn.Module):
    """Huber loss function wrapper."""

    def __init__(self, reduction: str = "mean", delta: float = 1.0):
        super().__init__()
        self.loss_fn = torch.nn.HuberLoss(reduction=reduction, delta=delta)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.loss_fn(pred, target)


# Auraloss time-domain functions
@register_function("esr_loss", differentiable=True)
class ESRLoss(torch.nn.Module):
    """Error-to-Signal Ratio loss wrapper."""

    def __init__(self, **kwargs):
        super().__init__()
        self.loss_fn = auraloss.time.ESRLoss(**kwargs)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.loss_fn(pred, target)


@register_function("dc_loss", differentiable=True)
class DCLoss(torch.nn.Module):
    """DC component loss wrapper."""

    def __init__(self, **kwargs):
        super().__init__()
        self.loss_fn = auraloss.time.DCLoss(**kwargs)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.loss_fn(pred, target)


@register_function("si_sdr_loss", differentiable=True)
class SISDRLoss(torch.nn.Module):
    """Scale-Invariant Source-to-Distortion Ratio loss wrapper."""

    def __init__(self, **kwargs):
        super().__init__()
        self.loss_fn = auraloss.time.SISDRLoss(**kwargs)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.loss_fn(pred, target)


# Additional time-domain metrics (non-differentiable)
@register_function("snr_metric", differentiable=False, requires_no_grad=True)
class SNRMetric(torch.nn.Module):
    """Signal-to-Noise Ratio metric."""

    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        noise = pred - target
        signal_power = torch.mean(target**2)
        noise_power = torch.mean(noise**2)
        snr = 10 * torch.log10(signal_power / (noise_power + self.eps))
        return snr


@register_function("thd_metric", differentiable=False, requires_no_grad=True)
class THDMetric(torch.nn.Module):
    """Total Harmonic Distortion metric (simplified)."""

    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # Simplified THD calculation based on RMS difference
        distortion = pred - target
        signal_rms = torch.sqrt(torch.mean(target**2))
        distortion_rms = torch.sqrt(torch.mean(distortion**2))
        thd = distortion_rms / (signal_rms + self.eps)
        return 100 * thd  # Return as percentage


@register_function("mape_loss", differentiable=True)
class MAPELoss(torch.nn.Module):
    """Mean Absolute Percentage Error loss wrapper.

    Useful for relative error measurement in audio modeling.
    """

    def __init__(self):
        super().__init__()
        self.mape = tm.MeanAbsolutePercentageError()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # Ensure metric is on the same device as input tensors
        if hasattr(self.mape, "to"):
            self.mape = self.mape.to(pred.device)

        # Handle batch dimensions - torchmetrics expects [N, ...] format
        original_shape = pred.shape
        if len(original_shape) > 2:
            pred = pred.view(-1, original_shape[-1])
            target = target.view(-1, original_shape[-1])
        elif len(original_shape) == 2 and original_shape[0] == 1:
            pred = pred.squeeze(0)
            target = target.squeeze(0)

        return self.mape(pred, target)


@register_function("cosine_similarity_loss", differentiable=True)
class CosineSimilarityLoss(torch.nn.Module):
    """Cosine Similarity loss wrapper.

    Measures the cosine of the angle between predicted and target vectors.
    Useful for measuring phase/alignment in audio signals.
    """

    def __init__(self, reduction="mean"):
        super().__init__()
        self.cos_sim = tm.CosineSimilarity(reduction=reduction)
        self.reduction = reduction

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # Ensure metric is on the same device as input tensors
        if hasattr(self.cos_sim, "to"):
            self.cos_sim = self.cos_sim.to(pred.device)

        # Handle batch dimensions - torchmetrics expects [N, ...] format
        original_shape = pred.shape
        if len(original_shape) > 2:
            pred = pred.view(-1, original_shape[-1])
            target = target.view(-1, original_shape[-1])
        elif len(original_shape) == 2 and original_shape[0] == 1:
            pred = pred.squeeze(0)
            target = target.squeeze(0)

        # Cosine similarity returns values in [-1, 1], convert to loss (lower is better)
        similarity = self.cos_sim(pred, target)
        return 1.0 - similarity  # Convert to loss: 0 = perfect match, 2 = opposite


@register_function("log_cosh_loss", differentiable=True)
class LogCoshLoss(torch.nn.Module):
    """Log-Cosh loss wrapper from auraloss.

    Logarithm of the hyperbolic cosine of the prediction error.
    Works like L2 for small errors and L1 for large errors.
    """

    def __init__(self, eps=1e-8):
        super().__init__()
        self.log_cosh = auraloss.time.LogCoshLoss(eps=eps)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.log_cosh(pred, target)

"""
Frequency Domain Evaluation Functions

This module registers frequency-domain evaluation functions from auraloss
that can be used as both losses and metrics.
"""

import torch
import auraloss
from ..registry import register_function


@register_function("stft_loss", differentiable=True)
class STFTLoss(torch.nn.Module):
    """Single-resolution STFT loss wrapper."""

    def __init__(self, **kwargs):
        super().__init__()
        self.loss_fn = auraloss.freq.STFTLoss(**kwargs)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # Handle tensor dimensions - auraloss expects 3D tensors [batch, channels, samples]
        if pred.dim() == 2:
            pred = pred.unsqueeze(1)  # Add channel dimension if missing
        if target.dim() == 2:
            target = target.unsqueeze(1)

        return self.loss_fn(pred, target)


@register_function("mrstft_loss", differentiable=True)
class MultiResolutionSTFTLoss(torch.nn.Module):
    """Multi-Resolution STFT loss wrapper."""

    def __init__(self, **kwargs):
        super().__init__()
        self.loss_fn = auraloss.freq.MultiResolutionSTFTLoss(**kwargs)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # Handle tensor dimensions - auraloss expects 3D tensors [batch, channels, samples]
        if pred.dim() == 2:
            pred = pred.unsqueeze(1)  # Add channel dimension if missing
        if target.dim() == 2:
            target = target.unsqueeze(1)

        return self.loss_fn(pred, target)


@register_function("melstft_loss", differentiable=True)
class MelSTFTLoss(torch.nn.Module):
    """Mel-Scale STFT loss wrapper."""

    def __init__(self, **kwargs):
        super().__init__()
        self.loss_fn = auraloss.freq.MelSTFTLoss(**kwargs)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # Handle tensor dimensions - auraloss expects 3D tensors [batch, channels, samples]
        if pred.dim() == 2:
            pred = pred.unsqueeze(1)  # Add channel dimension if missing
        if target.dim() == 2:
            target = target.unsqueeze(1)

        return self.loss_fn(pred, target)


@register_function("random_stft_loss", differentiable=True)
class RandomResolutionSTFTLoss(torch.nn.Module):
    """Random Resolution STFT loss wrapper."""

    def __init__(self, **kwargs):
        super().__init__()
        self.loss_fn = auraloss.freq.RandomResolutionSTFTLoss(**kwargs)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # Handle tensor dimensions - auraloss expects 3D tensors [batch, channels, samples]
        if pred.dim() == 2:
            pred = pred.unsqueeze(1)  # Add channel dimension if missing
        if target.dim() == 2:
            target = target.unsqueeze(1)

        return self.loss_fn(pred, target)

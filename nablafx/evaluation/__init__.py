"""
Evaluation Module

This module provides a unified evaluation system for audio effects modeling,
including a registry-based approach for losses and metrics, and flexible
loss composition utilities.
"""

# Import registry system
from .registry import EvaluationRegistry, register_function

# Import flexible loss system
from .flexible_loss import FlexibleLoss, FlexibleLossWithMetrics

# Import all evaluation functions to populate the registry
from . import functions


# Convenience functions
def list_available_losses():
    """List all available differentiable loss functions."""
    return FlexibleLoss.list_available_losses()


def list_available_metrics():
    """List all available metric functions (both differentiable and non-differentiable)."""
    return EvaluationRegistry.list_functions()


def get_function(name: str, **kwargs):
    """Get an evaluation function by name."""
    return EvaluationRegistry.get_function(name, **kwargs)


def get_function_info(name: str):
    """Get metadata about an evaluation function."""
    return EvaluationRegistry.get_function_info(name)


__all__ = [
    "EvaluationRegistry",
    "register_function",
    "FlexibleLoss",
    "FlexibleLossWithMetrics",
    "list_available_losses",
    "list_available_metrics",
    "get_function",
    "get_function_info",
]

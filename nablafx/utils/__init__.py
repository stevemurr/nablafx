"""
NablAFx Utilities Module

This module contains utility functions and classes for the nablafx package.

Modules:
- utilities: General utility classes like Rearrange and PTanh
- plotting: Plotting functions for model visualization and analysis
"""

# Import utility classes
from .utilities import Rearrange, PTanh

# Import plotting functions
from .plotting import (
    fig2img,
    plot_static_params,
    plot_phase_inv,
    plot_gain,
    plot_dc_offset,
    plot_parametric_eq,
    plot_lowpass,
    plot_highpass,
    plot_fir_filter,
    plot_static_mlp_nonlinearity,
    plot_static_rational_nonlinearity,
    plot_gb_model,
    plot_frequency_response_steps,
    plot_frequency_response,
)

__all__ = [
    # Utility classes
    "Rearrange",
    "PTanh",
    # Plotting functions
    "fig2img",
    "plot_static_params",
    "plot_phase_inv",
    "plot_gain",
    "plot_dc_offset",
    "plot_parametric_eq",
    "plot_lowpass",
    "plot_highpass",
    "plot_fir_filter",
    "plot_static_mlp_nonlinearity",
    "plot_static_rational_nonlinearity",
    "plot_gb_model",
    "plot_frequency_response_steps",
    "plot_frequency_response",
]

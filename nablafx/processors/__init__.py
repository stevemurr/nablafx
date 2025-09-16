"""
NablAFx Processors Module

This module contains all signal processing components for audio effects modeling,
including DSP utilities, differentiable DSP processors, and neural architectures.
"""

# DSP utilities
from .dsp import (
    denormalize,
    denormalize_parameters,
    fft_freqz,
    fft_sosfreqz,
    freqdomain_fir,
    lfilter_via_fsm,
    sosfilt_via_fsm,
    sosfilt,
    biquad,
)

# DDSP processors
from .ddsp import (
    PhaseShift,
    PhaseInversion,
    Gain,
    DCOffset,
    ParametricEQ,
    ShelvingEQ,
    Peaking,
    Lowpass,
    Highpass,
    Lowshelf,
    Highshelf,
    StaticFIRFilter,
    TanhNonlinearity,
    StaticMLPNonlinearity,
    StaticRationalNonlinearity,
)

# Neural architectures
from .lstm import LSTM
from .tcn import TCN
from .s4 import S4
from .gcn import GCN
from .siren import Sine, Siren, SirenNet, Modulator

__all__ = [
    # DSP utilities
    "denormalize",
    "denormalize_parameters",
    "fft_freqz",
    "fft_sosfreqz",
    "freqdomain_fir",
    "lfilter_via_fsm",
    "sosfilt_via_fsm",
    "sosfilt",
    "biquad",
    # DDSP processors
    "PhaseShift",
    "PhaseInversion",
    "Gain",
    "DCOffset",
    "ParametricEQ",
    "ShelvingEQ",
    "Peaking",
    "Lowpass",
    "Highpass",
    "Lowshelf",
    "Highshelf",
    "StaticFIRFilter",
    "TanhNonlinearity",
    "StaticMLPNonlinearity",
    "StaticRationalNonlinearity",
    # Neural architectures
    "LSTM",
    "TCN",
    "S4",
    "GCN",
    "Sine",
    "Siren",
    "SirenNet",
    "Modulator",
]

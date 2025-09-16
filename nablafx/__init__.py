"""
NablaFX: A modular framework for audio processing and neural networks.

This package provides:
- Data processing utilities and datasets
- Digital signal processing (DSP) tools
- Differentiable digital signal processing (DDSP) modules
- Neural network architectures for audio
- Training systems and controllers
- Core interfaces and model implementations
"""

# Core classes - available at top level for convenience
from .core import (
    # Models
    BlackBoxModel,
    GreyBoxModel,
    # Interfaces
    Processor,
    Controller,
    # Systems
    BaseSystem,
    BlackBoxSystem,
    BlackBoxSystemWithTBPTT,
    GreyBoxSystem,
    GreyBoxSystemWithTBPTT,
)

# Module-level imports for organized access
from . import data
from . import processors
from . import core
from . import callbacks
from . import controllers
from . import utils

__version__ = "0.1.0"

__all__ = [
    # Core classes
    "BlackBoxModel",
    "GreyBoxModel",
    "Processor",
    "Controller",
    "BaseSystem",
    "BlackBoxSystem",
    "BlackBoxSystemWithTBPTT",
    "GreyBoxSystem",
    "GreyBoxSystemWithTBPTT",
    # Modules
    "data",
    "processors",
    "core",
    "callbacks",
    "controllers",
    "utils",
]

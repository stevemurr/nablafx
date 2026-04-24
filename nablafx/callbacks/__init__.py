"""
NablAFx Callback System

This module contains callback classes for logging and monitoring during training.
These callbacks extract logging functionality from the system classes to provide
a modular, configurable logging architecture.

Available Callbacks:
- AudioLoggingCallback: For logging audio samples during training
- MetricsLoggingCallback: For computing and logging evaluation metrics
- FrequencyResponseCallback: For logging frequency response plots
- FADComputationCallback: For computing Fréchet Audio Distance metrics
- ParameterLoggingCallback: For logging gray-box model parameter plots
- AudioChainLoggingCallback: For logging audio at each processing block
- ConsoleProgressCallback: Prints train/val progress to stdout (TTY-agnostic)
"""

from .audio_logging import AudioLoggingCallback
from .metrics_logging import MetricsLoggingCallback
from .frequency_response import FrequencyResponseCallback
from .fad_computation import FADComputationCallback
from .parameter_logging import ParameterLoggingCallback
from .audio_chain_logging import AudioChainLoggingCallback
from .console_progress import ConsoleProgressCallback

__all__ = [
    "AudioLoggingCallback",
    "MetricsLoggingCallback",
    "FrequencyResponseCallback",
    "FADComputationCallback",
    "ParameterLoggingCallback",
    "AudioChainLoggingCallback",
    "ConsoleProgressCallback",
]

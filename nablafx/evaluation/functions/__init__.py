"""
Evaluation Functions Package

This package contains all evaluation functions organized by domain.
Importing any module will automatically register its functions with the registry.
"""

# Import all modules to register functions
from . import time_domain
from . import frequency_domain
from . import audio_specific

__all__ = ["time_domain", "frequency_domain", "audio_specific"]

"""
Core module containing fundamental classes for nablafx.

This module contains:
- Base interfaces and abstract classes (Processor, Controller)
- Model implementations (BlackBoxModel, GreyBoxModel)
- System implementations for training and inference
"""

from .interfaces import Processor, Controller
from .models import BlackBoxModel, GreyBoxModel
from .system import BaseSystem, BlackBoxSystem, BlackBoxSystemWithTBPTT, GreyBoxSystem, GreyBoxSystemWithTBPTT

__all__ = [
    # Interfaces
    "Processor",
    "Controller",
    # Models
    "BlackBoxModel",
    "GreyBoxModel",
    # Systems
    "BaseSystem",
    "BlackBoxSystem",
    "BlackBoxSystemWithTBPTT",
    "GreyBoxSystem",
    "GreyBoxSystemWithTBPTT",
]

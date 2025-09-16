"""
Controllers Package

This package contains different types of controllers for audio processing chains:

- Static controllers: Non-time-varying parameter control (DummyController, StaticController)
- Conditional controllers: Parameter control based on external conditions (StaticCondController, DynamicCondController)
- Dynamic controllers: Time-varying parameter control based on input signal (DynamicController)
"""

from .controllers import (
    DummyController,
    StaticController,
    StaticCondController,
    DynamicController,
    DynamicCondController,
)

__all__ = [
    "DummyController",
    "StaticController",
    "StaticCondController",
    "DynamicCondController",
    "DynamicController",
]

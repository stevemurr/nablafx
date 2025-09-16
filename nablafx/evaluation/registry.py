"""
Evaluation Function Registry

This module provides a unified registry for all evaluation functions that can be used
as both loss functions (for optimization) and metrics (for logging/evaluation).
"""

from typing import Dict, List, Type, Any, Optional
import torch
from abc import ABC, abstractmethod


class EvaluationFunction(ABC):
    """Base class for evaluation functions that can be used as losses or metrics."""

    @abstractmethod
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute the evaluation function."""
        pass

    @property
    @abstractmethod
    def is_differentiable(self) -> bool:
        """Whether this function is differentiable (can be used as loss)."""
        pass

    @property
    @abstractmethod
    def requires_no_grad(self) -> bool:
        """Whether this function should be computed with torch.no_grad()."""
        pass


class EvaluationRegistry:
    """Registry for all evaluation functions that can be used as losses or metrics."""

    _functions: Dict[str, Dict[str, Any]] = {}

    @classmethod
    def register(cls, name: str, differentiable: bool = True, requires_no_grad: bool = False):
        """
        Register an evaluation function.

        Args:
            name: Unique name for the function
            differentiable: Whether function is differentiable (can be used as loss)
            requires_no_grad: Whether function should be computed with no_grad
        """

        def decorator(func_class):
            if name in cls._functions:
                raise ValueError(f"Function '{name}' is already registered")

            cls._functions[name] = {"class": func_class, "differentiable": differentiable, "requires_no_grad": requires_no_grad}
            return func_class

        return decorator

    @classmethod
    def get_function(cls, name: str, **kwargs):
        """Get an instance of a registered function."""
        if name not in cls._functions:
            raise ValueError(f"Function '{name}' not registered. Available: {list(cls._functions.keys())}")
        return cls._functions[name]["class"](**kwargs)

    @classmethod
    def get_function_info(cls, name: str) -> Dict[str, Any]:
        """Get metadata about a registered function."""
        if name not in cls._functions:
            raise ValueError(f"Function '{name}' not registered")
        return cls._functions[name].copy()

    @classmethod
    def list_functions(cls, differentiable_only: bool = False) -> List[str]:
        """List all registered functions."""
        if differentiable_only:
            return [name for name, info in cls._functions.items() if info["differentiable"]]
        return list(cls._functions.keys())

    @classmethod
    def is_differentiable(cls, name: str) -> bool:
        """Check if a function is differentiable."""
        if name not in cls._functions:
            raise ValueError(f"Function '{name}' not registered")
        return cls._functions[name]["differentiable"]

    @classmethod
    def requires_no_grad(cls, name: str) -> bool:
        """Check if a function requires no_grad."""
        if name not in cls._functions:
            raise ValueError(f"Function '{name}' not registered")
        return cls._functions[name]["requires_no_grad"]


def register_function(name: str, differentiable: bool = True, requires_no_grad: bool = False):
    """Convenience decorator for registering evaluation functions."""
    return EvaluationRegistry.register(name, differentiable, requires_no_grad)

"""
FlexibleLoss System

This module provides a registry-based loss composition system that allows
dynamic creation of loss functions from configuration files using the
evaluation function registry.
"""

import torch
from typing import List, Dict, Any, Union
from .registry import EvaluationRegistry


class FlexibleLoss(torch.nn.Module):
    """
    Registry-based flexible loss function that combines multiple loss functions with weights.

    This class uses the evaluation function registry to dynamically instantiate loss functions
    by name, making it easy to compose complex loss functions from configuration files.

    Args:
        losses: List of loss configurations, each containing:
            - 'name': The registered name of the loss function
            - 'weight': The weight for this loss function
            - 'params': Optional parameters for the loss function
            - 'alias': Optional alias for logging purposes (defaults to name)

    Example:
        losses = [
            {'name': 'l1_loss', 'weight': 1.0},
            {'name': 'edc_loss', 'weight': 0.5, 'params': {'eps': 1e-6}},
            {'name': 'mrstft_loss', 'weight': 1.0, 'params': {'fft_sizes': [1024, 2048]}}
        ]
        loss_fn = FlexibleLoss(losses)
    """

    def __init__(self, losses: List[Dict[str, Any]]) -> None:
        super().__init__()

        # Import evaluation functions to ensure registry is populated
        from . import functions

        if not losses:
            raise ValueError("At least one loss function must be provided")

        self.loss_functions = torch.nn.ModuleList()
        self.weights = []
        self.names = []
        self.aliases = []

        for i, loss_config in enumerate(losses):
            if not isinstance(loss_config, dict):
                raise ValueError(f"Loss config {i} must be a dictionary")

            if "name" not in loss_config:
                raise ValueError(f"Loss config {i} must contain 'name' key")

            if "weight" not in loss_config:
                raise ValueError(f"Loss config {i} must contain 'weight' key")

            loss_name = loss_config["name"]
            weight = loss_config["weight"]
            params = loss_config.get("params", {})
            alias = loss_config.get("alias", loss_name)

            # Validate that the loss function is registered and differentiable
            if not EvaluationRegistry.is_differentiable(loss_name):
                raise ValueError(f"Loss function '{loss_name}' is not differentiable and cannot be used as a loss")

            # Get loss function from registry
            try:
                loss_fn = EvaluationRegistry.get_function(loss_name, **params)
            except ValueError as e:
                available_losses = [name for name in EvaluationRegistry.list_functions() if EvaluationRegistry.is_differentiable(name)]
                raise ValueError(
                    f"Loss function '{loss_name}' not found in registry. " f"Available differentiable losses: {available_losses}"
                ) from e

            if not isinstance(weight, (int, float)):
                raise ValueError(f"Weight for loss {i} must be a number")

            self.loss_functions.append(loss_fn)
            self.weights.append(float(weight))
            self.names.append(loss_name)
            self.aliases.append(alias)

    def forward(self, pred: torch.Tensor, target: torch.Tensor):
        """
        Compute weighted sum of all loss functions.

        Args:
            pred: Predicted audio tensor
            target: Target audio tensor

        Returns:
            If only one loss function: returns the weighted loss value
            If multiple loss functions: returns tuple of (individual_losses..., total_loss)
        """
        individual_losses = []
        total_loss = 0.0

        for loss_fn, weight in zip(self.loss_functions, self.weights):
            loss_value = loss_fn(pred, target)
            weighted_loss = weight * loss_value
            individual_losses.append(weighted_loss)
            total_loss += weighted_loss

        # Return format consistent with existing WeightedMultiLoss
        if len(individual_losses) == 1:
            return individual_losses[0]
        else:
            return tuple(individual_losses + [total_loss])

    def get_loss_names(self) -> List[str]:
        """Return list of loss function names for logging purposes."""
        return self.names

    def get_loss_aliases(self) -> List[str]:
        """Return list of loss function aliases for logging purposes."""
        return self.aliases

    def get_weights(self) -> List[float]:
        """Return list of weights for each loss function."""
        return self.weights

    def get_registry_info(self) -> Dict[str, Dict[str, Any]]:
        """Return registry information for all loss functions."""
        info = {}
        for name in self.names:
            info[name] = EvaluationRegistry.get_function_info(name)
        return info

    @classmethod
    def list_available_losses(cls) -> List[str]:
        """List all available differentiable loss functions from the registry."""
        # Import evaluation functions to ensure registry is populated
        from . import functions

        return [name for name in EvaluationRegistry.list_functions() if EvaluationRegistry.is_differentiable(name)]

    @classmethod
    def from_config_dict(cls, config: Dict[str, Any]) -> "FlexibleLoss":
        """
        Create FlexibleLoss from a configuration dictionary.

        Args:
            config: Dictionary with 'losses' key containing list of loss configs

        Example:
            config = {
                'losses': [
                    {'name': 'l1_loss', 'weight': 1.0},
                    {'name': 'edc_loss', 'weight': 0.5, 'params': {'eps': 1e-6}}
                ]
            }
            loss_fn = FlexibleLoss.from_config_dict(config)
        """
        if "losses" not in config:
            raise ValueError("Config must contain 'losses' key")

        return cls(config["losses"])


class FlexibleLossWithMetrics(FlexibleLoss):
    """
    Extended FlexibleLoss that also computes metrics alongside losses.

    This allows you to compute both differentiable losses (for training) and
    non-differentiable metrics (for evaluation) in a single forward pass.
    """

    def __init__(self, losses: List[Dict[str, Any]], metrics: List[Dict[str, Any]] = None) -> None:
        super().__init__(losses)

        self.metric_functions = torch.nn.ModuleList()
        self.metric_names = []
        self.metric_aliases = []

        if metrics:
            for i, metric_config in enumerate(metrics):
                if not isinstance(metric_config, dict):
                    raise ValueError(f"Metric config {i} must be a dictionary")

                if "name" not in metric_config:
                    raise ValueError(f"Metric config {i} must contain 'name' key")

                metric_name = metric_config["name"]
                params = metric_config.get("params", {})
                alias = metric_config.get("alias", metric_name)

                # Get metric function from registry (can be differentiable or not)
                try:
                    metric_fn = EvaluationRegistry.get_function(metric_name, **params)
                except ValueError as e:
                    available_metrics = EvaluationRegistry.list_functions()
                    raise ValueError(
                        f"Metric function '{metric_name}' not found in registry. " f"Available functions: {available_metrics}"
                    ) from e

                self.metric_functions.append(metric_fn)
                self.metric_names.append(metric_name)
                self.metric_aliases.append(alias)

    def compute_metrics(self, pred: torch.Tensor, target: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute all metrics.

        Args:
            pred: Predicted audio tensor
            target: Target audio tensor

        Returns:
            Dictionary mapping metric names to computed values
        """
        metrics = {}

        for metric_fn, name, alias in zip(self.metric_functions, self.metric_names, self.metric_aliases):
            # Use no_grad for metrics that require it
            if EvaluationRegistry.requires_no_grad(name):
                with torch.no_grad():
                    metric_value = metric_fn(pred, target)
            else:
                metric_value = metric_fn(pred, target)

            metrics[alias] = metric_value

        return metrics

    def get_metric_names(self) -> List[str]:
        """Return list of metric function names."""
        return self.metric_names

    def get_metric_aliases(self) -> List[str]:
        """Return list of metric function aliases."""
        return self.metric_aliases

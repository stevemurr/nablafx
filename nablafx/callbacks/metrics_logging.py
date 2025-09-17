"""
Metrics Logging Callback

Handles computation and logging of various metrics during training.
Extracted from system.py to provide modular, configurable metrics logging.
Now uses the evaluation registry for flexible metric configuration.
"""

import torch
from typing import Any, Dict, List, Optional, Union
import lightning as pl
from lightning.pytorch.callbacks import Callback
from nablafx.evaluation.registry import EvaluationRegistry


class MetricsLoggingCallback(Callback):
    """
    Callback for computing and logging metrics during training.

    This callback leverages the evaluation registry to provide flexible,
    configurable metrics computation and logging.

    Args:
        metrics: List of metrics to compute. Each can be:
                - str: metric name from registry (e.g., "snr_metric")
                - dict: {"name": "metric_name", "alias": "custom_alias", "params": {...}}
        log_on_step: Whether to log metrics on each step (default: False)
        log_on_epoch: Whether to log metrics on each epoch (default: True)
        sync_dist: Whether to sync metrics across distributed processes (default: True)
        prefix: Prefix for metric names in logs (default: "metric")
    """

    def __init__(
        self,
        metrics: Optional[List[Union[str, Dict[str, Any]]]] = None,
        log_on_step: bool = False,
        log_on_epoch: bool = True,
        sync_dist: bool = True,
        prefix: str = "metric",
    ):
        super().__init__()
        self.log_on_step = log_on_step
        self.log_on_epoch = log_on_epoch
        self.sync_dist = sync_dist
        self.prefix = prefix

        # Initialize metrics using registry
        self.metrics = self._create_metrics_from_registry(metrics or self._get_default_metrics())

    def _get_default_metrics(self) -> List[str]:
        """Get default metrics configuration."""
        return ["snr_metric", "thd_metric", "zero_crossing_rate_metric"]

    def _create_metrics_from_registry(self, metrics_config: List[Union[str, Dict[str, Any]]]) -> Dict[str, Any]:
        """Create metrics using the evaluation registry."""
        metrics = {}

        for metric_spec in metrics_config:
            if isinstance(metric_spec, str):
                # Simple string reference
                name = metric_spec
                alias = name.replace("_metric", "").replace("_loss", "")
                params = {}
            elif isinstance(metric_spec, dict):
                # Full configuration
                name = metric_spec["name"]
                alias = metric_spec.get("alias", name.replace("_metric", "").replace("_loss", ""))
                params = metric_spec.get("params", {})
            else:
                raise ValueError(f"Invalid metric specification: {metric_spec}")

            # Validate it's a metric (not differentiable, requires no_grad)
            if name not in EvaluationRegistry.list_functions():
                raise ValueError(f"Unknown metric: {name}")

            if EvaluationRegistry.is_differentiable(name):
                print(f"Warning: {name} is differentiable - typically used as loss, not metric")

            # Create metric function
            metric_fn = EvaluationRegistry.get_function(name, **params)
            metrics[alias] = {"function": metric_fn, "requires_no_grad": EvaluationRegistry.requires_no_grad(name)}

        return metrics

    def on_train_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
    ) -> None:
        """Compute metrics on training batch end."""
        if self.log_on_step:
            self._compute_and_log_metrics(trainer, pl_module, batch, "train")

    def on_validation_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        """Compute metrics on validation batch end."""
        self._compute_and_log_metrics(trainer, pl_module, batch, "val")

    def on_test_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        """Compute metrics on test batch end."""
        self._compute_and_log_metrics(trainer, pl_module, batch, "test")

    def _compute_and_log_metrics(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        batch: Any,
        mode: str,
    ) -> None:
        """Compute and log metrics for the given batch."""
        # Extract batch data
        if hasattr(pl_module.model, "num_controls") and pl_module.model.num_controls > 0:
            input_audio, target = batch[:2]
            controls = batch[2] if len(batch) > 2 else None
        else:
            input_audio, target = batch[:2]
            controls = None

        # Generate predictions
        pl_module.model.reset_states()
        if mode == "train" and hasattr(pl_module.model, "detach_states"):
            pl_module.model.detach_states()

        with torch.no_grad():
            pred = pl_module(input_audio, controls, train=(mode == "train"))

        # Compute metrics using registry
        for alias, metric_info in self.metrics.items():
            try:
                metric_fn = metric_info["function"]
                requires_no_grad = metric_info["requires_no_grad"]

                # Prepare tensors
                pred_for_metric = pred.clone()
                target_for_metric = target.clone()

                # Ensure metric function is on the same device as the tensors
                if hasattr(metric_fn, "to"):
                    metric_fn = metric_fn.to(pred_for_metric.device)

                # Compute metric with appropriate gradient context
                if requires_no_grad:
                    with torch.no_grad():
                        metric_value = metric_fn(pred_for_metric, target_for_metric)
                else:
                    metric_value = metric_fn(pred_for_metric, target_for_metric)

                # Clean up tensors
                pred_for_metric = pred_for_metric.detach().cpu()
                target_for_metric = target_for_metric.detach().cpu()

                # Ensure metric value is a scalar tensor
                if isinstance(metric_value, tuple):
                    metric_value = metric_value[0]  # Take first element if tuple
                if hasattr(metric_value, "detach"):
                    metric_value = metric_value.detach().cpu()

                # Log metric
                pl_module.log(
                    f"{self.prefix}/{mode}/{alias}",
                    metric_value,
                    on_step=self.log_on_step,
                    on_epoch=self.log_on_epoch,
                    prog_bar=False,
                    logger=True,
                    sync_dist=self.sync_dist,
                )

                # Reset metric state if possible (for stateful metrics)
                if hasattr(metric_fn, "reset"):
                    metric_fn.reset()

            except Exception as e:
                # Log error but don't crash training
                print(f"Warning: Failed to compute metric {alias}: {e}")
                continue

        # Cleanup
        del pred, target, input_audio
        if controls is not None:
            del controls

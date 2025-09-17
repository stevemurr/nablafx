"""
FAD Computation Callback

Handles computation and logging of Fréchet Audio Distance (FAD) scores.
Extracted from system.py to provide modular, configurable FAD computation.
Now uses the evaluation registry for flexible FAD model configuration.
"""

import os
import torch
import wandb
from typing import Any, Dict, List, Optional, Union
import lightning as pl
from lightning.pytorch.callbacks import Callback
from nablafx.evaluation.registry import EvaluationRegistry


class FADComputationCallback(Callback):
    """
    Callback for computing and logging Fréchet Audio Distance (FAD) scores.

    This callback leverages the evaluation registry to provide flexible,
    configurable FAD computation using different pre-trained models.

    Args:
        compute_on_train_end: Whether to compute FAD at training end (default: True)
        compute_on_test_end: Whether to compute FAD at test end (default: True)
        compute_every_n_epochs: Compute FAD every N epochs during validation (default: None)
        fad_metrics: List of FAD metrics to compute. Each can be:
                    - str: metric name from registry (e.g., "fad_vggish_metric")
                    - dict: {"name": "fad_metric_name", "alias": "custom_alias", "params": {...}}
        checkpoint_dir: Directory containing FAD model checkpoints (default: "checkpoints_fad")
        prefix: Prefix for FAD metric names in logs (default: "fad")
    """

    def __init__(
        self,
        compute_on_train_end: bool = True,
        compute_on_test_end: bool = True,
        compute_every_n_epochs: Optional[int] = None,
        fad_metrics: Optional[List[Union[str, Dict[str, Any]]]] = None,
        checkpoint_dir: str = "checkpoints_fad",
        prefix: str = "fad",
    ):
        super().__init__()
        self.compute_on_train_end = compute_on_train_end
        self.compute_on_test_end = compute_on_test_end
        self.compute_every_n_epochs = compute_every_n_epochs
        self.checkpoint_dir = checkpoint_dir
        self.prefix = prefix

        # Initialize FAD metrics using registry
        self.fad_metrics = self._create_fad_metrics_from_registry(fad_metrics or self._get_default_fad_metrics())

    def _get_default_fad_metrics(self) -> List[str]:
        """Get default FAD metrics configuration.

        Includes metrics supported by the current frechet_audio_distance package.
        Currently supports: VGGish, PANN, and CLAP models.
        AFX-Rep is not supported by the current package version.
        """
        return ["fad_vggish_metric", "fad_pann_metric", "fad_clap_metric"]

    def _create_fad_metrics_from_registry(self, metrics_config: List[Union[str, Dict[str, Any]]]) -> Dict[str, Any]:
        """Create FAD metrics using the evaluation registry."""
        metrics = {}

        for metric_spec in metrics_config:
            if isinstance(metric_spec, str):
                # Simple string reference
                name = metric_spec
                alias = name.replace("_metric", "").replace("fad_", "")
                params = {}
            elif isinstance(metric_spec, dict):
                # Full configuration
                name = metric_spec["name"]
                alias = metric_spec.get("alias", name.replace("_metric", "").replace("fad_", ""))
                params = metric_spec.get("params", {})
            else:
                raise ValueError(f"Invalid FAD metric specification: {metric_spec}")

            # Validate it's a FAD metric
            if name not in EvaluationRegistry.list_functions():
                raise ValueError(f"Unknown FAD metric: {name}")

            if not name.startswith("fad_"):
                print(f"Warning: {name} doesn't appear to be a FAD metric")

            # Add checkpoint directory to params if not specified
            if "ckpt_dir" not in params:
                params["ckpt_dir"] = self.checkpoint_dir

            # Create metric function
            metric_fn = EvaluationRegistry.get_function(name, **params)
            metrics[alias] = {"function": metric_fn, "requires_no_grad": EvaluationRegistry.requires_no_grad(name)}

        return metrics

    def on_train_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """Compute FAD at training end."""
        if self.compute_on_train_end:
            self._compute_and_log_fad_registry(trainer, pl_module, "train_end")

    def on_test_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """Compute FAD at test end."""
        if self.compute_on_test_end:
            self._compute_and_log_fad_registry(trainer, pl_module, "test")

    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """Compute FAD every N validation epochs."""
        if self.compute_every_n_epochs is not None and trainer.current_epoch % self.compute_every_n_epochs == 0:
            self._compute_and_log_fad_registry(trainer, pl_module, "val")

    def _compute_and_log_fad_registry(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        phase: str,
    ) -> None:
        """Compute and log FAD scores using registry-based metrics."""
        if not hasattr(trainer.logger, "experiment"):
            return  # Skip if no wandb logger

        try:
            print(f"\nComputing and logging FAD using registry for {phase}...")

            # Get audio directories
            run_dir = trainer.logger.experiment.dir
            pred_dir = os.path.join(run_dir, f"media/audio/audio/{phase}/pred")
            target_dir = os.path.join(run_dir, f"media/audio/audio/{phase}/target")

            # Check if directories exist
            if not os.path.exists(pred_dir) or not os.path.exists(target_dir):
                print(f"Warning: Audio directories not found for FAD computation in {phase}")
                return

            print(f"Computing FAD for {run_dir}...")

            # Compute FAD scores using registry metrics
            fad_scores = {}
            for alias, metric_info in self.fad_metrics.items():
                try:
                    print(f"Computing FAD score using {alias}...")
                    metric_fn = metric_info["function"]
                    requires_no_grad = metric_info["requires_no_grad"]

                    # Compute FAD score with appropriate gradient context
                    if requires_no_grad:
                        with torch.no_grad():
                            fad_score = metric_fn(target_dir, pred_dir)
                    else:
                        fad_score = metric_fn(target_dir, pred_dir)

                    # Extract scalar value if needed
                    if isinstance(fad_score, tuple):
                        fad_score = fad_score[0]
                    if hasattr(fad_score, "item"):
                        fad_score = fad_score.item()

                    fad_scores[alias] = fad_score
                    print(f"FAD score for {alias}: {fad_score:.4f}")

                    # Log to wandb
                    if hasattr(trainer.logger, "experiment"):
                        wandb.log({f"{self.prefix}/{phase}/{alias}": fad_score})

                except Exception as e:
                    print(f"Warning: Failed to compute FAD for {alias}: {e}")
                    continue

            # Log summary if we have scores
            if fad_scores:
                avg_fad = sum(fad_scores.values()) / len(fad_scores)
                print(f"Average FAD score for {phase}: {avg_fad:.4f}")
                if hasattr(trainer.logger, "experiment"):
                    wandb.log({f"{self.prefix}/{phase}/average": avg_fad})

        except Exception as e:
            print(f"Error computing FAD for {phase}: {e}")
            return

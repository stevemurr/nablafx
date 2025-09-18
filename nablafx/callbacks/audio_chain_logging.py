"""
Audio Chain Logging Callback

Handles logging of audio output at each block in gray-box models.
Logs audio at each processing stage for debugging and analysis.
"""

import torch
import wandb
from typing import Any, Optional
import lightning as pl
from lightning.pytorch.callbacks import Callback


class AudioChainLoggingCallback(Callback):
    """
    Callback for logging audio output at each block in gray-box models.

    This callback captures and logs the audio signal as it passes through
    each processing block in gray-box models, useful for debugging and
    understanding the effect of each processing stage.

    Args:
        log_on_train_start: Whether to log audio chain at training start (default: False)
        log_on_validation: Whether to log audio chain during validation (default: True)
        log_on_test: Whether to log audio chain during testing (default: True)
        log_test_batches: Number of test batches to log (default: 5)
        max_samples_per_batch: Maximum samples to log per batch (default: 3)
        sample_rate: Sample rate for audio logging (default: 48000)
    """

    def __init__(
        self,
        log_on_train_start: bool = False,
        log_on_validation: bool = True,
        log_on_test: bool = True,
        log_test_batches: int = 5,
        max_samples_per_batch: int = 3,
        sample_rate: int = 48000,
    ):
        super().__init__()
        self.log_on_train_start = log_on_train_start
        self.log_on_validation = log_on_validation
        self.log_on_test = log_on_test
        self.log_test_batches = log_test_batches
        self.max_samples_per_batch = max_samples_per_batch
        self.sample_rate = sample_rate

    def on_train_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """Log audio chain at training start."""
        if self.log_on_train_start and self._is_greybox_model(pl_module):
            try:
                # Get a batch from the dataloader
                batch = next(iter(trainer.train_dataloader))

                # Move batch to device
                if hasattr(pl_module.model, "num_controls") and pl_module.model.num_controls > 0:
                    input_audio, target, controls = batch
                    controls = controls.to(pl_module.device)
                else:
                    input_audio, target = batch
                    controls = None

                input_audio = input_audio.to(pl_module.device)

                # Log audio chain
                pl_module.model.reset_states()
                self._log_audio_at_each_block(trainer, pl_module, input_audio, controls, "train_start")

            except Exception as e:
                print(f"Warning: Failed to log audio chain at train start: {e}")

    def on_validation_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        """Log audio chain during validation."""
        if self.log_on_validation and batch_idx == 0 and self._is_greybox_model(pl_module):

            # Extract batch data
            if hasattr(pl_module.model, "num_controls") and pl_module.model.num_controls > 0:
                input_audio, target, controls = batch
            else:
                input_audio, target = batch
                controls = None

            self._log_audio_at_each_block(trainer, pl_module, input_audio, controls, "val")

    def on_test_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        """Log audio chain during testing."""
        if self.log_on_test and batch_idx < self.log_test_batches and self._is_greybox_model(pl_module):

            # Extract batch data
            if hasattr(pl_module.model, "num_controls") and pl_module.model.num_controls > 0:
                input_audio, target, controls = batch
            else:
                input_audio, target = batch
                controls = None

            self._log_audio_at_each_block(trainer, pl_module, input_audio, controls, "test")

    def _is_greybox_model(self, pl_module: pl.LightningModule) -> bool:
        """Check if the model is a gray-box model."""
        return (
            hasattr(pl_module.model, "processor")
            and hasattr(pl_module.model, "controller")
            and hasattr(pl_module.model.processor, "processors")
            and hasattr(pl_module.model.controller, "controllers")
        )

    def _log_audio_at_each_block(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        input_audio: torch.Tensor,
        controls: Optional[torch.Tensor],
        mode: str,
    ) -> None:
        """
        Log audio output at each block in the processing chain.
        """
        if not hasattr(trainer.logger, "experiment"):
            return  # Skip if no wandb logger

        try:
            print(f"\nLogging audio at each block for {mode}...")

            x = input_audio.to(pl_module.device)
            y = [input_audio.detach().cpu()]  # Store audio at each stage (start with input)

            # Get control parameters
            control_params = pl_module.model.controller(x, controls if controls is not None else None)

            # Process through each block
            for processor, ctrl_params in zip(pl_module.model.processor.processors, control_params):
                if ctrl_params is not None:
                    ctrl_params = ctrl_params.to(pl_module.device)

                with torch.no_grad():
                    y_i, _ = processor(x, ctrl_params, train=False)

                y.append(y_i.detach().cpu())
                x = y_i  # Use GPU tensor for next iteration

            # Log audio for each block and each batch item
            num_samples = min(len(input_audio), self.max_samples_per_batch)

            for batch_idx in range(num_samples):
                for block_idx, block_audio in enumerate(y):
                    try:
                        audio_np = block_audio[batch_idx].numpy()
                        if audio_np.ndim > 1:
                            audio_np = audio_np[0, :]  # Take first channel

                        # Create descriptive caption
                        if block_idx == 0:
                            caption = f"input_sample_{batch_idx}"
                        else:
                            caption = f"block_{block_idx-1}_output_sample_{batch_idx}"

                        trainer.logger.experiment.log(
                            {
                                f"audio/chain/{mode}/sample_{batch_idx}/block_{block_idx}": wandb.Audio(
                                    audio_np, self.sample_rate, caption=caption
                                )
                            },
                            step=trainer.global_step,
                        )

                    except Exception as e:
                        print(f"Warning: Failed to log audio for sample {batch_idx}, block {block_idx}: {e}")
                        continue

            print(f"Successfully logged audio chain for {mode} ({num_samples} samples, {len(y)} blocks)")

        except Exception as e:
            print(f"Warning: Failed to log audio at each block for {mode}: {e}")

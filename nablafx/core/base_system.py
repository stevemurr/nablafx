import os
import torch
import random
import auraloss
import sys

# import pytorch_lightning as pl
import lightning as pl
import torchmetrics as tm
import wandb

from typing import List, Optional
from nablafx.utils.plotting import plot_frequency_response_steps
from nablafx.evaluation.flexible_loss import FlexibleLoss
from frechet_audio_distance import FrechetAudioDistance

module_path = os.path.abspath(os.path.join(".."))
if module_path not in sys.path:
    sys.path.append(module_path)


class BaseSystem(pl.LightningModule):
    def __init__(
        self,
        loss: torch.nn.Module,
        lr: float = 1e-4,
        log_media_every_n_steps: int = 10000,
        use_callbacks: bool = False,
    ):
        """
        Base system for audio effect modeling.

        Args:
            loss: Loss function to use
            lr: Learning rate
            log_media_every_n_steps: Steps between media logging (ignored if use_callbacks=True)
            use_callbacks: If True, relies on callbacks for logging instead of built-in methods.
                          When True, the following methods become no-ops:
                          - compute_and_log_metrics
                          - log_audio
                          - log_frequency_response
                          - compute_and_log_fad
        """
        super().__init__()
        self.loss = loss
        self.lr = lr
        self.log_media_every_n_steps = log_media_every_n_steps
        self.log_media_counter = 0
        self.log_input_and_target_flag = True
        self.use_callbacks = use_callbacks

        # metrics (only used if not using callbacks)
        if not self.use_callbacks:
            # WARNING: These are legacy hardcoded metrics for backward compatibility.
            # For new projects, use MetricsLoggingCallback with the evaluation registry
            # which provides better configurability and supports more metrics.
            self.metrics = {
                "mae": tm.MeanAbsoluteError(),
                "mape": tm.MeanAbsolutePercentageError(),
                "mse": tm.MeanSquaredError(),
                "cossim": tm.CosineSimilarity(),
                "logcosh": auraloss.time.LogCoshLoss(),
                "esr": auraloss.time.ESRLoss(),
                "dcloss": auraloss.time.DCLoss(),
            }
        else:
            self.metrics = {}
            print("📋 Using callback-based logging - built-in logging methods disabled")

    def forward(self, input: torch.Tensor, params: torch.Tensor, train: bool = False):
        return self.model(input, params, train=train)

    def common_step(self, batch, batch_idx, mode="train"):
        raise NotImplementedError

    def training_step(self, batch, batch_idx):
        loss = self.common_step(batch, batch_idx, mode="train")
        return loss

    def validation_step(self, batch, batch_idx):
        raise NotImplementedError

    def test_step(self, batch, batch_idx):
        raise NotImplementedError

    def on_train_start(self):
        # log gradients (always enabled)
        wandb.watch(self.model, log_freq=100)
        # Frequency response logging handled by callbacks if use_callbacks=True
        if not self.use_callbacks:
            pass  # self.log_frequency_response() # atm needs too much memory

    def on_before_optimizer_step(self, optimizer):
        # Compute the 2-norm for each layer (always enabled)
        # If using mixed precision, the gradients are already unscaled here
        norms = pl.pytorch.utilities.grad_norm(self, norm_type=2)
        self.log_dict(norms)

    def on_train_end(self):
        # FAD computation handled by callbacks if use_callbacks=True
        if not self.use_callbacks:
            # self.log_frequency_response() # atm needs too much memory
            self.compute_and_log_fad(mode="val")

    def on_test_epoch_end(self):
        # Logging handled by callbacks if use_callbacks=True
        if not self.use_callbacks:
            self.log_frequency_response()
            self.compute_and_log_fad(mode="test")

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.lr,
            betas=(0.9, 0.999),
            eps=1e-8,
        )

        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=20, verbose=True)

        return [optimizer], [{"scheduler": lr_scheduler, "monitor": "loss/val/tot", "interval": "epoch", "frequency": 1}]

    def compute_and_log_loss(self, pred, target, mode, should_log=True):

        losses = self.loss(pred, target)

        # Handle FlexibleLoss format
        if isinstance(self.loss, FlexibleLoss):
            if isinstance(losses, tuple):
                # Multiple losses: (loss1, loss2, ..., total_loss)
                individual_losses = losses[:-1]
                tot_loss = losses[-1]
            else:
                # Single loss
                individual_losses = [losses]
                tot_loss = losses

            # Compute scaled losses for logging (unweighted values)
            scaled_losses = []

            # Use aliases for loss names
            loss_names = self.loss.get_loss_aliases()
            weights = self.loss.get_weights()

            for loss_val, weight in zip(individual_losses, weights):
                if weight > 0:
                    scaled_losses.append(loss_val / weight)
                else:
                    scaled_losses.append(loss_val)

            tot_loss_scaled = sum(scaled_losses)

        else:
            # Simple loss function (single value)
            if isinstance(losses, (tuple, list)):
                tot_loss = sum(losses)
                individual_losses = list(losses)
            else:
                tot_loss = losses
                individual_losses = [losses]
            loss_names = ["loss"]
            scaled_losses = individual_losses
            tot_loss_scaled = tot_loss

        if should_log:
            # Log total loss
            self.log(
                f"loss/{mode}/tot",
                tot_loss,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                logger=True,
                sync_dist=True,
            )

            # Log individual losses
            for loss_val, loss_name in zip(individual_losses, loss_names):
                self.log(
                    f"loss/{mode}/{loss_name}",
                    loss_val,
                    on_step=False,
                    on_epoch=True,
                    prog_bar=False,
                    logger=True,
                    sync_dist=True,
                )

            # Log scaled losses (unweighted)
            self.log(
                f"loss_scaled/{mode}/tot",
                tot_loss_scaled,
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                logger=True,
                sync_dist=True,
            )

            for scaled_loss, loss_name in zip(scaled_losses, loss_names):
                self.log(
                    f"loss_scaled/{mode}/{loss_name}",
                    scaled_loss,
                    on_step=False,
                    on_epoch=True,
                    prog_bar=False,
                    logger=True,
                    sync_dist=True,
                )

        if torch.isnan(tot_loss):
            print("NaN loss encountered. Exiting...")
            sys.exit()

        return tot_loss

    def compute_and_log_metrics(self, pred, target, mode):
        """Compute and log metrics. No-op if using callbacks.

        WARNING: This is legacy code for backward compatibility.
        For new projects, use MetricsLoggingCallback with the evaluation registry
        which provides better configurability and supports more metrics including
        audio-specific ones like SNR, THD, zero-crossing rate, etc.
        """
        if self.use_callbacks:
            return  # Metrics handled by MetricsLoggingCallback

        for name, metric in self.metrics.items():
            metric = metric.to(self.device)
            if pred.dim() == 3:
                pred = pred.squeeze(1)
                target = target.squeeze(1)
            metric_value = metric(pred, target)

            metric_value = metric_value.detach().cpu()
            pred = pred.detach().cpu()
            target = target.detach().cpu()

            self.log(
                f"metrics/{mode}/{name}",
                metric_value,
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                logger=True,
                sync_dist=True,
            )
            if hasattr(metric, "reset"):
                metric.reset()
        pred, target, metric_value = None, None, None

    def log_audio(self, batch_idx, input, target, pred, mode):
        """Log audio samples. No-op if using callbacks.

        WARNING: This is legacy code for backward compatibility.
        For new projects, use AudioLoggingCallback which provides better
        configurability and more efficient audio logging.
        """
        if self.use_callbacks:
            return  # Audio logging handled by AudioLoggingCallback

        input, target, pred = input.detach().cpu(), target.detach().cpu(), pred.detach().cpu()

        if mode == "test":  # log all audio in test mode
            if batch_idx < 10:  # log only first 10 batches
                print(f"\nLogging audio for {mode} batch {batch_idx}...")
                for i, _ in enumerate(input):
                    self.logger.experiment.log(
                        {
                            f"audio/{mode}/input/b{batch_idx}-{i}": wandb.Audio(
                                input[i].cpu().numpy()[0, :], 48000, caption=f"input_b{batch_idx}-{i}"
                            ),
                            f"audio/{mode}/target/b{batch_idx}-{i}": wandb.Audio(
                                target[i].cpu().numpy()[0, :], 48000, caption=f"target_b{batch_idx}-{i}"
                            ),
                            f"audio/{mode}/pred/b{batch_idx}-{i}": wandb.Audio(
                                pred[i].cpu().numpy()[0, :], 48000, caption=f"pred_b{batch_idx}-{i}"
                            ),
                        },
                        step=self.trainer.global_step,
                    )
        elif self.log_input_and_target_flag:  # log input and target only once
            for i, _ in enumerate(input):
                self.logger.experiment.log(
                    {
                        f"audio/{mode}/input/b{batch_idx}-{i}": wandb.Audio(
                            input[i].cpu().numpy()[0, :], 48000, caption=f"input_b{batch_idx}-{i}"
                        ),
                        f"audio/{mode}/target/b{batch_idx}-{i}": wandb.Audio(
                            target[i].cpu().numpy()[0, :], 48000, caption=f"target_b{batch_idx}-{i}"
                        ),
                        f"audio/{mode}/pred/b{batch_idx}-{i}": wandb.Audio(
                            pred[i].cpu().numpy()[0, :], 48000, caption=f"pred_b{batch_idx}-{i}"
                        ),
                    },
                    step=self.trainer.global_step,
                )
            self.log_input_and_target_flag = False
        else:
            for i, _ in enumerate(input):
                self.logger.experiment.log(
                    {
                        f"audio/{mode}/pred/b{batch_idx}-{i}": wandb.Audio(
                            pred[i].cpu().numpy()[0, :], 48000, caption=f"pred_b{batch_idx}-{i}"
                        ),
                    },
                    step=self.trainer.global_step,
                )
        input, target, pred = None, None, None

    def log_frequency_response(self):
        """Log frequency response plot. No-op if using callbacks."""
        if self.use_callbacks:
            return  # Frequency response handled by FrequencyResponseCallback

        print("\nLogging frequency response...")
        self.model.reset_states()
        with torch.no_grad():
            plot = plot_frequency_response_steps(self.model)
        self.logger.experiment.log({f"response/freq+phase": [wandb.Image(plot, caption=f"")]}, step=self.trainer.global_step)

    def compute_and_log_fad(self, mode):
        """Compute and log FAD scores. No-op if using callbacks.

        WARNING: This is legacy code for backward compatibility.
        For new projects, use FADComputationCallback with the evaluation registry
        which provides better configurability and supports the latest models.
        """
        if self.use_callbacks:
            return  # FAD computation handled by FADComputationCallback

        print("\nComputing and logging FAD...")
        run_dir = self.logger.experiment.dir
        pred_dir = os.path.join(run_dir, f"media/audio/audio/{mode}/pred")
        target_dir = os.path.join(run_dir, f"media/audio/audio/{mode}/target")

        parent_dir = os.path.abspath(os.getcwd())
        ckpt_dir = os.path.join(parent_dir, "checkpoints_fad")

        print(f"\n\nComputing FAD for {run_dir}...")

        fad_vggish = FrechetAudioDistance(
            os.path.join(ckpt_dir, "vggish"),
            model_name="vggish",
            sample_rate=16000,
            use_pca=False,
            use_activation=False,
            verbose=False,
        )

        fad_score_vggish = fad_vggish.score(
            target_dir,
            pred_dir,
        )

        fad_pann = FrechetAudioDistance(
            os.path.join(ckpt_dir, "pann"),
            model_name="pann",
            sample_rate=32000,
            verbose=False,
        )

        fad_score_pann = fad_pann.score(
            target_dir,
            pred_dir,
        )

        fad_clap = FrechetAudioDistance(
            os.path.join(ckpt_dir, "clap"),
            model_name="clap",
            submodel_name="630k-audioset",
            sample_rate=48000,
            verbose=False,
            enable_fusion=False,
        )

        fad_score_clap = fad_clap.score(
            target_dir,
            pred_dir,
        )

        # AFX-Rep is not supported by the current frechet_audio_distance package
        # fad_afxrep = FrechetAudioDistance(
        #     os.path.join(ckpt_dir, "afx-rep"),
        #     model_name="afx-rep",
        #     sample_rate=48000,
        #     verbose=False,
        # )
        #
        # fad_score_afxrep = fad_afxrep.score(
        #     target_dir,
        #     pred_dir,
        # )

        print(f"\nFAD score (vggish): {fad_score_vggish}")
        print(f"FAD score (pann): {fad_score_pann}")
        print(f"FAD score (clap): {fad_score_clap}")
        # print(f"FAD score (afx-rep): {fad_score_afxrep}")

        self.logger.experiment.log(
            {
                f"metrics/{mode}/fad-vggish": fad_score_vggish,
                f"metrics/{mode}/fad-pann": fad_score_pann,
                f"metrics/{mode}/fad-clap": fad_score_clap,
                # f"metrics/{mode}/fad-afxrep": fad_score_afxrep,
            },
            step=self.trainer.global_step,
        )

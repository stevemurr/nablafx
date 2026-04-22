"""
End-to-end training smoke test against the 610B MicPreamp dataset.

Runs a tiny TCN (with and without rational activations) and an LSTM for a
handful of steps with Lightning's Trainer:
  - real paired data (DRY -> 610B target) via PluginDataset
  - FlexibleLoss (L1 + MRSTFT)
  - CUDA if available
  - checks loss actually decreases and training runs on the expected device

Invoke with:
    uv run python scripts/smoketest_train.py
"""

from __future__ import annotations

import tempfile
import time
from pathlib import Path

import lightning as pl
import torch

# Importing nablafx applies the rational-activations config patch.
from nablafx import BlackBoxModel, BlackBoxSystem
from nablafx.data import DryWetFilesPluginDataModule
from nablafx.evaluation.flexible_loss import FlexibleLoss
from nablafx.processors.lstm import LSTM
from nablafx.processors.tcn import TCN


DRY_DIR = Path("/home/murr/jupyter-redux/datasets/DRY")
WET_DIR = Path("/home/murr/jupyter-redux/datasets/610B-MicPreamp")
WET_SUBDIR = "G10dB_L6_ILine_H0dB_L0dB"
SAMPLE_RATE = 44100
SAMPLE_LENGTH = 22050  # 0.5 s
BATCH_SIZE = 8
MAX_STEPS = 30


class _StepLossRecorder(pl.pytorch.callbacks.Callback):
    """Grab the live training loss each step. The built-in loggers in nablafx
    use on_epoch=True, which doesn't flush often enough for a short smoke
    test to see progress.
    """

    def __init__(self):
        self.losses: list[float] = []
        self.training_device: str = ""

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        loss = outputs["loss"] if isinstance(outputs, dict) and "loss" in outputs else outputs
        if loss is not None:
            self.losses.append(float(loss.detach().cpu()))
        if not self.training_device:
            self.training_device = str(next(pl_module.parameters()).device)


def _make_datamodule() -> DryWetFilesPluginDataModule:
    assert DRY_DIR.exists(), f"missing dry root {DRY_DIR}"
    assert WET_DIR.exists(), f"missing wet root {WET_DIR}"
    return DryWetFilesPluginDataModule(
        root_dir_dry=str(DRY_DIR / "trainval"),
        root_dir_wet=str(WET_DIR / "trainval" / WET_SUBDIR),
        data_to_use=1.0,
        trainval_split=0.85,
        sample_length=SAMPLE_LENGTH,
        sample_rate=SAMPLE_RATE,
        preload=False,
        batch_size=BATCH_SIZE,
        num_workers=2,
    )


def _make_loss() -> FlexibleLoss:
    return FlexibleLoss(
        losses=[
            {"name": "l1_loss", "weight": 0.5, "alias": "l1"},
            {"name": "mrstft_loss", "weight": 0.5, "alias": "mrstft"},
        ]
    )


def _run(label: str, processor: torch.nn.Module) -> None:
    print(f"\n========== {label} ==========")
    dm = _make_datamodule()
    system = BlackBoxSystem(
        model=BlackBoxModel(processor=processor),
        loss=_make_loss(),
        lr=5e-3,
        # use_callbacks=True bypasses legacy wandb-specific audio/FAD hooks
        # inside BaseSystem. No callbacks are registered, so nothing is logged
        # beyond what the step recorder captures — fine for a smoke test.
        use_callbacks=True,
    )

    recorder = _StepLossRecorder()
    with tempfile.TemporaryDirectory() as logdir:
        trainer = pl.Trainer(
            accelerator="gpu" if torch.cuda.is_available() else "cpu",
            devices=1,
            max_steps=MAX_STEPS,
            logger=pl.pytorch.loggers.CSVLogger(save_dir=logdir, name="smoke"),
            callbacks=[recorder],
            enable_progress_bar=False,
            enable_checkpointing=False,
            num_sanity_val_steps=0,
            # Legacy BlackBoxSystem.validation_step calls self.logger.experiment.log
            # which is wandb-specific, so skip validation in the smoke test.
            limit_val_batches=0,
            limit_train_batches=MAX_STEPS + 1,
            deterministic=False,
            benchmark=True,
        )

        start = time.time()
        trainer.fit(system, datamodule=dm)
        elapsed = time.time() - start

    if torch.cuda.is_available():
        assert recorder.training_device.startswith("cuda"), (
            f"{label}: expected training on cuda, got {recorder.training_device!r}"
        )

    losses = recorder.losses
    assert len(losses) >= MAX_STEPS // 2, (
        f"{label}: expected ~{MAX_STEPS} losses, got {len(losses)}"
    )

    third = max(1, len(losses) // 3)
    first_mean = sum(losses[:third]) / third
    last_mean = sum(losses[-third:]) / third
    print(
        f"[smoke] {label}: device={recorder.training_device} "
        f"first-third mean={first_mean:.4f} last-third mean={last_mean:.4f} "
        f"walltime={elapsed:.1f}s"
    )
    assert last_mean < first_mean, f"{label}: loss did not decrease ({first_mean} -> {last_mean})"


def main() -> int:
    cases = [
        ("TCN tanh", TCN(
            num_inputs=1, num_outputs=1, num_blocks=4, kernel_size=5,
            dilation_growth=2, channel_width=8, stack_size=4,
            causal=True, residual=True, act_type="tanh",
        )),
        ("TCN rational", TCN(
            num_inputs=1, num_outputs=1, num_blocks=3, kernel_size=5,
            dilation_growth=2, channel_width=8, stack_size=3,
            causal=True, residual=True, act_type="rational",
        )),
        ("LSTM", LSTM(
            num_inputs=1, num_outputs=1, hidden_size=16, num_layers=1,
            residual=True,
        )),
    ]
    for label, proc in cases:
        _run(label, proc)

    print("\n[smoke] OK — all cases passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

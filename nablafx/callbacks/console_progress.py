"""Console progress callback.

Lightning's default ``TQDMProgressBar`` auto-disables in non-TTY environments
(jupyter kernels, containers without ``-it``), so a container run shows
nothing on stdout between ``trainer.fit`` and completion. This callback
prints a line every N train steps with the latest loss + LR; always works
regardless of terminal capabilities.
"""

from __future__ import annotations

import time

import lightning.pytorch as pl


class ConsoleProgressCallback(pl.Callback):
    """Print step / loss / LR every ``every_n_steps`` training steps.

    Also emits a one-line summary at the end of each validation run. Writes
    to stdout with ``flush=True`` so the output isn't held in buffers when
    piped or captured.
    """

    def __init__(self, every_n_steps: int = 50) -> None:
        super().__init__()
        self.every_n_steps = max(1, int(every_n_steps))
        self._last_print_time: float = 0.0

    def on_train_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        self._last_print_time = time.monotonic()
        total = trainer.max_steps if trainer.max_steps > 0 else trainer.max_epochs
        unit = "steps" if trainer.max_steps > 0 else "epochs"
        print(f"[train] starting: target={total} {unit}", flush=True)

    def on_train_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs,
        batch,
        batch_idx: int,
    ) -> None:
        step = trainer.global_step
        if step == 0 or step % self.every_n_steps != 0:
            return
        metrics = trainer.logged_metrics
        loss = metrics.get("loss/train/tot")
        loss_val = float(loss) if loss is not None else float("nan")
        lr = trainer.optimizers[0].param_groups[0]["lr"]
        now = time.monotonic()
        steps_per_sec = self.every_n_steps / max(1e-6, now - self._last_print_time)
        self._last_print_time = now
        target = trainer.max_steps if trainer.max_steps > 0 else -1
        print(
            f"[train] step {step:>7d}/{target} "
            f"loss={loss_val:.4f} lr={lr:.2e} {steps_per_sec:.1f} it/s",
            flush=True,
        )

    def on_validation_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        metrics = trainer.logged_metrics
        loss = metrics.get("loss/val/tot")
        if loss is None:
            return
        print(f"[val]   step {trainer.global_step:>7d} loss={float(loss):.4f}", flush=True)

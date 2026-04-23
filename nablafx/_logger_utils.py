"""Small helpers for logger-aware code paths.

Several legacy code paths (in core/base_system.py, core/greybox_system.py, and
a handful of callbacks) call `self.logger.experiment.log(...)` or
`wandb.Audio(...)` / `wandb.Image(...)` directly. That's fine when the
trainer is wired to a WandbLogger; it crashes under CSV/TensorBoard/None.

`is_wandb_logger` gives each call site a single guard to skip silently when
wandb isn't actually in play.
"""

from __future__ import annotations


def is_wandb_logger(logger) -> bool:
    """True iff `logger` is a live WandbLogger with an initialized run."""
    if logger is None:
        return False
    try:
        import wandb
        from lightning.pytorch.loggers import WandbLogger
    except Exception:
        return False
    return isinstance(logger, WandbLogger) and wandb.run is not None

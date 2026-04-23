"""
Hydra entrypoint for nablafx training.

Usage:
    uv run python scripts/train.py data=<data> model=<model> [trainer=<trainer>] [overrides...]

Examples:
    uv run python scripts/train.py data=610b_trainval model=tcn/model_bb_tcn-45-s-16
    uv run python scripts/train.py -m data=610b_trainval model=tcn/model_bb_tcn-45-s-16,lstm/model_bb_lstm-32

See conf/ for the config tree. Callbacks are composed in conf/trainer/*.yaml
via Hydra's defaults list — toggle from the CLI with `~trainer.callbacks.<role>`
or `+trainer/callbacks@trainer.callbacks.<role>=<name>`.
"""

from __future__ import annotations

import os

import hydra
import torch
from hydra.utils import instantiate
from lightning import seed_everything
from omegaconf import DictConfig, OmegaConf

import nablafx  # noqa: F401  (applies rational-activations config patch at import)


@hydra.main(version_base="1.3", config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    seed_everything(42, workers=True)
    torch.set_float32_matmul_precision("high")
    torch.use_deterministic_algorithms(False, warn_only=True)
    os.environ.setdefault("WANDB__SERVICE_WAIT", "300")

    datamodule = instantiate(cfg.data, _convert_="all")
    system = instantiate(cfg.model, _convert_="all")

    # Callbacks live in a dict keyed by role so CLI overrides can add/remove
    # individual ones. Lightning's Trainer wants a list, so unpack here.
    trainer_cfg = OmegaConf.to_container(cfg.trainer, resolve=True)
    callback_specs = trainer_cfg.pop("callbacks", {}) or {}
    logger_spec = trainer_cfg.pop("logger", None)
    target = trainer_cfg.pop("_target_")

    callbacks = [instantiate(c) for c in callback_specs.values()]
    logger = instantiate(logger_spec) if logger_spec else None

    trainer = hydra.utils.get_class(target)(callbacks=callbacks, logger=logger, **trainer_cfg)

    trainer.fit(system, datamodule=datamodule)


if __name__ == "__main__":
    main()

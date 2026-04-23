"""
Hydra entrypoint for nablafx fit / validate / test.

Usage:
    uv run python scripts/train.py data=<data> model=<model> [mode=fit|test|validate] [ckpt_path=...]

Examples (fit):
    uv run python scripts/train.py data=1176LN-Limiter_trainval model=tcn/model_bb_tcn-45-s-16

Example (test): load the resolved config of a trained run, override data, supply ckpt.
    uv run python scripts/train.py \\
      --config-path ../outputs/2026-04-22/17-00-00/.hydra --config-name config \\
      mode=test \\
      data=1176LN-Limiter_test \\
      ckpt_path=outputs/2026-04-22/17-00-00/logs/version_0/checkpoints/last.ckpt

Rebase dataset location for a run (see `dataset_root` in conf/config.yaml):
    uv run python scripts/train.py data=1176LN-Limiter_trainval model=tcn/model_bb_tcn-45-s-16 dataset_root=/shared/datasets

Multirun sweep:
    uv run python scripts/train.py -m data=1176LN-Limiter_trainval model=tcn/bb_tcn-45-s-16,lstm/bb_lstm-32

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

    callbacks = [instantiate(c, _convert_="all") for c in callback_specs.values()]
    logger = instantiate(logger_spec, _convert_="all") if logger_spec else None

    trainer = hydra.utils.get_class(target)(callbacks=callbacks, logger=logger, **trainer_cfg)

    mode = cfg.get("mode", "fit")
    ckpt_path = cfg.get("ckpt_path", None)

    if mode == "fit":
        trainer.fit(system, datamodule=datamodule, ckpt_path=ckpt_path)
    elif mode == "validate":
        if not ckpt_path:
            raise ValueError("mode=validate requires ckpt_path=<path>")
        trainer.validate(system, datamodule=datamodule, ckpt_path=ckpt_path)
    elif mode == "test":
        if not ckpt_path:
            raise ValueError("mode=test requires ckpt_path=<path>")
        trainer.test(system, datamodule=datamodule, ckpt_path=ckpt_path)
    else:
        raise ValueError(f"unknown mode {mode!r}; expected fit | validate | test")


if __name__ == "__main__":
    main()

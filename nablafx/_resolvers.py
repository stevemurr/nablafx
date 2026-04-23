"""
Custom OmegaConf resolvers used by nablafx configs.

Registered at package import time so they're available when Hydra composes
the top-level config in nablafx.__main__.
"""

from __future__ import annotations

from omegaconf import OmegaConf


_DATA_SUFFIXES = ("_trainval", "_test", "_val")


def _dataset_from_data_choice(data_choice: str) -> str:
    """Return the dataset slug for a data config name.

    `data=1176LN-Limiter_trainval` -> `1176LN-Limiter`. Used by hydra.run.dir
    so train and test runs for the same effect land under one folder.
    """
    for suffix in _DATA_SUFFIXES:
        if data_choice.endswith(suffix):
            return data_choice[: -len(suffix)]
    return data_choice


def apply() -> None:
    OmegaConf.register_new_resolver(
        "nablafx.dataset_from_data_choice",
        _dataset_from_data_choice,
        replace=True,
    )

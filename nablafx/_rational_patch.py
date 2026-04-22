"""
Redirect rational-activations' init-weight lookup to the extended config
shipped inside nablafx.

The upstream wheel bundles a `rationals_config.json` that only contains
A5/4, B5/4, C5/4, D5/4, N5/4. nablafx uses additional entries (A4/3, A6/5,
A2/1, A3/2) across its processors, so we maintain an extended copy as
package data and patch `rational.utils.get_weights.get_parameters` to read
from it. Rational-activations keeps working unmodified everywhere else.
"""

from __future__ import annotations

import json
from importlib.resources import files
from pathlib import Path


_CONFIG_RESOURCE = files(__package__).joinpath("rationals_config.json")


def _load_config() -> dict:
    with _CONFIG_RESOURCE.open("r") as f:
        return json.load(f)


def _patched_get_parameters(rational_version: str, degrees, approx_func: str):
    nd, dd = degrees
    full_name = f"Rational_version_{rational_version}{nd}/{dd}"
    cfg = _load_config()
    if full_name not in cfg or approx_func not in cfg[full_name]:
        raise KeyError(
            f"{full_name} approximating {approx_func!r} not found in nablafx's "
            f"rationals_config.json (at {_CONFIG_RESOURCE})."
        )
    params = cfg[full_name][approx_func]
    return params["init_w_numerator"], params["init_w_denominator"]


def apply() -> None:
    """Monkey-patch rational.utils.get_weights.get_parameters.

    Also rebinds the already-imported name inside rational.torch.rationals,
    which does `from rational.utils.get_weights import get_parameters` at
    module load time.
    """
    import rational.utils.get_weights as _gw

    _gw.get_parameters = _patched_get_parameters

    try:
        import rational.torch.rationals as _rt

        _rt.get_parameters = _patched_get_parameters
    except ImportError:
        pass

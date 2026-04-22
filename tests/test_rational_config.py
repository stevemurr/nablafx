"""
Smoke test: rational-activations must find init weights for the degree
combinations nablafx actually uses, which live in nablafx's extended
rationals_config.json (not the upstream default).
"""

import pytest
import torch


@pytest.fixture(autouse=True, scope="module")
def _apply_patch():
    # Importing nablafx applies the rational-activations monkey-patch.
    import nablafx  # noqa: F401


@pytest.mark.parametrize(
    "degrees",
    [(4, 3), (6, 5), (5, 4), (3, 2), (2, 1)],
)
def test_rational_construct_and_forward(degrees):
    from rational.torch import Rational

    act = Rational(approx_func="tanh", degrees=list(degrees), version="A")
    device = next(act.parameters()).device
    x = torch.linspace(-3.0, 3.0, 128, device=device)
    y = act(x)
    assert y.shape == x.shape
    assert torch.isfinite(y).all()


def test_config_shipped_in_package():
    from importlib.resources import files

    path = files("nablafx").joinpath("rationals_config.json")
    assert path.is_file()


def test_extended_keys_present():
    import json
    from importlib.resources import files

    with files("nablafx").joinpath("rationals_config.json").open() as f:
        cfg = json.load(f)

    # These are the entries upstream does NOT ship — they must be in ours.
    for key in ("Rational_version_A4/3", "Rational_version_A6/5"):
        assert key in cfg, f"missing {key} in nablafx rationals_config.json"
        assert "tanh" in cfg[key]


def test_tcn_block_rational_forward():
    """TCN conditional block uses Rational(degrees=[4,3]) — needs the extended
    A4/3 config entry. Also confirms end-to-end works on whatever device the
    module lands on.
    """
    from nablafx.processors.blocks import TCNCondBlock

    blk = TCNCondBlock(
        in_ch=1, out_ch=4, causal=True, batchnorm=False, residual=False,
        kernel_size=3, padding=0, dilation=1, groups=1, bias=True,
        cond_type=None, cond_dim=0, cond_block_size=128, cond_num_layers=1,
        act_type="rational",
    )
    if torch.cuda.is_available():
        blk = blk.cuda()
    device = next(blk.parameters()).device
    x = torch.randn(2, 1, 100, device=device)
    y = blk(x)
    assert y.shape[0] == 2 and y.shape[1] == 4
    assert torch.isfinite(y).all()

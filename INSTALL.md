# NablaFX Installation Guide

NablaFX uses [uv](https://docs.astral.sh/uv/) for environment and dependency
management. If you don't have uv installed yet:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

## Install

```bash
git clone https://github.com/mcomunita/nablafx.git
cd nablafx
uv sync
```

That's it. `uv sync` creates a `.venv`, resolves all dependencies, and
installs `nablafx` in editable mode.

To run anything inside the environment, prefix commands with `uv run` (or
activate `.venv` directly):

```bash
uv run python -c "import nablafx; print(nablafx.__version__)"
uv run pytest tests/test_rational_config.py
```

## GPU / CUDA support

`pyproject.toml` is configured to pull `torch`, `torchaudio`, and
`torchvision` from the PyTorch **CUDA 13** wheel index
(`https://download.pytorch.org/whl/cu130`). This gives native support for
Blackwell GPUs (e.g. DGX Spark GB10, sm_121) out of the box.

To target a different CUDA version (or CPU-only), edit the
`[[tool.uv.index]]` block in `pyproject.toml`:

```toml
[[tool.uv.index]]
name = "pytorch-cu130"
url = "https://download.pytorch.org/whl/cu130"   # or cu124, cu121, cpu, ...
explicit = true
```

...then re-run `uv sync`.

## Notes on `rational-activations`

`rational-activations==0.2.0` is a regular dependency, but its wheel
declares a stale `torch==1.7.1` pin in its metadata. `pyproject.toml`
includes a uv-level override (`[tool.uv] override-dependencies`) that
drops that pin during resolution. No manual `--no-deps` gymnastics are
needed.

The wheel is pure-Python: its CUDA sources are unused at runtime and it
falls back to a PyTorch-only implementation, so it works on any modern
torch/CUDA combination (including CUDA 13 / Blackwell).

nablafx ships an extended `rationals_config.json` as package data (it
contains the `A4/3`, `A6/5`, `A2/1`, `A3/2` entries that several nablafx
processors rely on but upstream does not ship). On `import nablafx`, the
`rational.utils.get_weights.get_parameters` loader is transparently
redirected to that file — rational-activations still works normally
elsewhere.

## Verify the install

```bash
uv run pytest tests/test_rational_config.py -v
```

The smoke test builds a Rational activation at every degree configuration
nablafx uses, runs a forward pass through a `TCNCondBlock(act_type="rational")`,
and checks package-data shipping. If it passes, install is healthy.

## Troubleshooting

**Torch can't find my GPU.** Check driver/CUDA version match:
```bash
nvidia-smi
uv run python -c "import torch; print(torch.version.cuda, torch.cuda.is_available())"
```
If `torch.version.cuda` is older than your driver supports, confirm the
`pytorch-cu130` index in `pyproject.toml` matches your CUDA runtime.

**Clean reinstall.**
```bash
rm -rf .venv uv.lock
uv sync
```

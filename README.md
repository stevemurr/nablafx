<div align="center">
<img width="250px" src="https://raw.githubusercontent.com/mcomunita/nablafx/master/assets/nablafx_1.png">
<br><br>
  
# NablAFx

**differentiable black-box and gray-box audio effects modeling framework**

[Paper](https://arxiv.org/abs/2502.11668)

[Marco Comunità](https://mcomunita.github.io/), [Christian J. Steinmetz](https://www.christiansteinmetz.com/), [Joshua D. Reiss](http://www.eecs.qmul.ac.uk/~josh/)

Centre for Digital Music, Queen Mary University of London, UK<br>

</div>

## Abstract
We present NablAFx, an open-source framework developed to support research in differentiable black-box and gray-box modeling of audio effects. 
Built in PyTorch, NablAFx offers a versatile ecosystem to configure, train, evaluate, and compare various architectural approaches. 
It includes classes to manage model architectures, datasets, and training, along with features to compute and log losses, metrics and media, and plotting functions to facilitate detailed analysis. 
It incorporates implementations of established black-box architectures and conditioning methods, as well as differentiable DSP blocks and controllers, enabling the creation of both parametric and non-parametric gray-box signal chains.

## Citation

```BibTex
@misc{comunità2025nablafxframeworkdifferentiableblackbox,
      title={NablAFx: A Framework for Differentiable Black-box and Gray-box Modeling of Audio Effects}, 
      author={Marco Comunità and Christian J. Steinmetz and Joshua D. Reiss},
      year={2025},
      eprint={2502.11668},
      archivePrefix={arXiv},
      primaryClass={cs.SD},
      url={https://arxiv.org/abs/2502.11668}, 
}
```

```BibTex
@misc{comunità2025differentiableblackboxgrayboxmodeling,
      title={Differentiable Black-box and Gray-box Modeling of Nonlinear Audio Effects}, 
      author={Marco Comunità and Christian J. Steinmetz and Joshua D. Reiss},
      year={2025},
      eprint={2502.14405},
      archivePrefix={arXiv},
      primaryClass={cs.SD},
      url={https://arxiv.org/abs/2502.14405}, 
}
```

## Support

To show your support please consider giving this repo a star :star:. Thanks! :metal:

## Installation

NablaFX uses [uv](https://docs.astral.sh/uv/) for environment management:

```bash
# Install uv if needed
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone and sync
git clone https://github.com/mcomunita/nablafx.git
cd nablafx
uv sync
```

That's it. All dependencies (including `rational-activations` and a CUDA-13
PyTorch build for native Blackwell / DGX Spark support) are resolved from
`pyproject.toml`.

Run any command inside the project environment with `uv run`:

```bash
uv run python -c "import nablafx; print(nablafx.__version__)"
uv run pytest tests/test_rational_config.py
```

See [INSTALL.md](INSTALL.md) for CUDA-version overrides and troubleshooting.

## Setup for Training

### Data

NablAFx is setup to work with the [ToneTwist AFx Dataset](https://github.com/mcomunita/tonetwist-afx-dataset), and so are the dataset classes provided in `./nablafx/data/`, but you can also write your own custom class and use a different dataset.

Make data folder and either move data to that folder or create a symbolic link:

```bash
mkdir data
cd data
ln -s /path/to/TONETWIST-AFX-DATASET/
```

### Logs

Runs land in `outputs/<date>/<time>/` (or `multirun/<date>/<time>/<idx>/` for
sweeps) — Hydra creates these automatically and dumps the resolved config
under `.hydra/config.yaml` for each run. CSV logging is on by default and
writes to `logs/` inside the run directory.

To use Weights & Biases instead:

```bash
wandb login
uv run python scripts/train.py ... trainer/logger@trainer.logger=wandb \
  trainer.logger.entity=YOUR_USER trainer.logger.name=my-run
```

### Frechét Audio Distance checkpoints

The checkpoints to compute Frechét Audio Distance need to be in specific folders:

```bash
mkdir checkpoints_fad
```

The first time you will compute FAD (which is at test time) the checkpoints are automatically downloaded and placed into subfolders.

## Train

Training is driven by [Hydra](https://hydra.cc/). Configs live in `conf/`,
organised into groups:

```
conf/
  config.yaml           # top-level defaults + Hydra settings
  data/*.yaml           # data modules (one per dataset/pedal)
  model/<arch>/*.yaml   # model variants grouped by architecture
  trainer/              # trainers + callback / logger subgroups
```

Pick one of each from the CLI:

```bash
uv run python scripts/train.py \
  data=data_ampeg-optocomp_trainval \
  model=s4/model_bb_s4-b4-s4-c16 \
  trainer=bb
```

Override any field by name (use `+key=value` when the key isn't already in
the config schema):

```bash
uv run python scripts/train.py \
  data=data_klon-centaur_trainval \
  model=tcn/model_bb_tcn-45-s-16 \
  trainer=bb \
  model.lr=1e-4 \
  trainer.max_steps=50000
```

Toggle individual trainer callbacks:

```bash
# disable one:
  '~trainer.callbacks.early_stopping'
# enable an opt-in:
  '+trainer/callbacks@trainer.callbacks.fad=fad'
# swap logger to W&B:
  trainer/logger@trainer.logger=wandb trainer.logger.entity=YOUR_USER
```

Sweeps run with `-m` (Cartesian product):

```bash
uv run python scripts/train.py -m \
  data=data_klon-centaur_trainval,data_ibanez-ts9_trainval \
  model=tcn/model_bb_tcn-45-s-16,lstm/model_bb_lstm-32 \
  model.lr=1e-3,1e-4
```

Example wrappers: `scripts/train_bb.sh`, `scripts/train_gb.sh`.

## Test

The `test` subcommand has not yet been ported from the old Lightning CLI
entrypoint. See `scripts/test.sh` for the current workaround and tracking.

## Params, FLOPs, MACs and Real-time factor:

Scripts to measure all these metrics of a model are in the `benchmarks` folder. The code is not very organized and beautiful at the moment but does the job.

## Pre-train MLP Nonlinearity or FIR

The script to pretrain differentiable processors (see `nablafx/processors/ddsp.py`) like a `StaticMLPNonlinearity` or a `StaticFIRFilter` is `nablafx/scripts/pretrain.py`.
Examples of pretrained nonlinearities and filters are in `nablafx/weights`.

## Contributions

There is a lot that can be done to improve and expand NablAFx, and we encourage anyone to contribute or suggest new features.

## Credits

* LSTM - [https://github.com/Alec-Wright/Automated-GuitarAmpModelling](https://github.com/Alec-Wright/Automated-GuitarAmpModelling)
* TCN - [https://github.com/csteinmetz1/micro-tcn](https://github.com/csteinmetz1/micro-tcn)
* GCN - [https://github.com/mcomunita/gcn-tfilm](https://github.com/mcomunita/gcn-tfilm)
* S4 - [https://github.com/int0thewind/s4-dynamic-range-compressor](https://github.com/int0thewind/s4-dynamic-range-compressor)
* dasp - [https://github.com/csteinmetz1/dasp-pytorch](https://github.com/csteinmetz1/dasp-pytorch)
* rational activations - [https://github.com/ml-research/rational_activations](https://github.com/ml-research/rational_activations)
* auraloss - [https://github.com/csteinmetz1/auraloss](https://github.com/csteinmetz1/auraloss)

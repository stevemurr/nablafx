#!/usr/bin/env bash
# Testing example. The Hydra entrypoint currently only runs `fit`; for `test`
# the pattern is to hold the trained checkpoint + its resolved config and
# invoke `trainer.test(system, datamodule=datamodule, ckpt_path=...)` from
# a small script. This is a TODO — the Lightning CLI `test` subcommand is
# not yet reimplemented in the Hydra entrypoint.
#
# For now, to evaluate a checkpoint:
#
# CUDA_VISIBLE_DEVICES=0 \
# uv run python scripts/train.py \
#   --config-path outputs/<timestamp>/.hydra \
#   --config-name config \
#   data=data-param_multidrive-ffuzz_test \
#   +ckpt_path=outputs/<timestamp>/logs/version_0/checkpoints/last.ckpt
#
# (assumes you extend scripts/train.py to honour `ckpt_path`; see README).
echo "test flow not yet ported to Hydra entrypoint — see comments" >&2
exit 1

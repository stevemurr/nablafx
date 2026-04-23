#!/usr/bin/env bash
# Evaluate a trained checkpoint on a test set.
#
# Specify the same MODEL + TRAINER that were used for training (the Lightning
# checkpoint stores weights only — you re-construct the architecture via
# Hydra instantiation, then load the weights in via ckpt_path=).
#
# Usage:
#   CKPT=outputs/.../checkpoints/last.ckpt \
#   MODEL=tcn/model_bb_tcn-45-s-16 \
#   TEST_DATA=610b_test \
#   bash scripts/test.sh

set -euo pipefail
cd "$(dirname "$0")/.."

CKPT="${CKPT:?set CKPT=<path/to/checkpoint.ckpt>}"
MODEL="${MODEL:?set MODEL=<group/path> e.g. tcn/model_bb_tcn-45-s-16}"
TEST_DATA="${TEST_DATA:-610b_test}"
TRAINER="${TRAINER:-bb}"

CUDA_VISIBLE_DEVICES=0 \
uv run python scripts/train.py \
  mode=test \
  "data=${TEST_DATA}" \
  "model=${MODEL}" \
  "trainer=${TRAINER}" \
  "ckpt_path=${CKPT}"

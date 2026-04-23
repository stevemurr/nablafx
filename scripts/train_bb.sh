#!/usr/bin/env bash
# Black-box training example. Pick any data/model/trainer from `conf/`.
# Override any field with `key=value` (append with `+key=value` if it's not
# in the schema). Multirun with `-m` runs the cartesian product.
CUDA_VISIBLE_DEVICES=0 \
uv run python scripts/train.py \
  data=data_ampeg-optocomp_trainval \
  model=s4/model_bb_s4-b4-s4-c16 \
  trainer=bb

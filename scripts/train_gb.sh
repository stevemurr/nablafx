#!/usr/bin/env bash
# Gray-box training example.
CUDA_VISIBLE_DEVICES=1 \
uv run nablafx \
  data=data_ampeg-optocomp_trainval \
  model=gb/gb_comp/model_gb_comp_peq.s+g.d+peq.s+g.s \
  trainer=gb

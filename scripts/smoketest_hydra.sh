#!/usr/bin/env bash
# End-to-end Hydra smoke test — one run per architecture family against
# the 610B-MicPreamp trainval set. Each run trains for a few steps to confirm
# the Hydra entrypoint, config composition, instantiation, and GPU placement
# all work.
#
# Usage:
#   uv run bash scripts/smoketest_hydra.sh
#
# Exits non-zero if any architecture fails.

set -euo pipefail

cd "$(dirname "$0")/.."

# Shared overrides for all cases: skip validation (legacy wandb-only path) and
# strip callbacks that need training to converge or that write to wandb.
COMMON=(
  data=610B-MicPreamp_trainval
  trainer=bb
  trainer.max_steps=5
  trainer.benchmark=false
  +trainer.limit_val_batches=0
  '~trainer.callbacks.audio'
  '~trainer.callbacks.metrics'
  '~trainer.callbacks.early_stopping'
  '~trainer.callbacks.checkpoint'
)

run_one() {
  local label="$1"
  local model="$2"
  echo ""
  echo "========== ${label} =========="
  uv run nablafx "${COMMON[@]}" "model=${model}"
}

run_one "TCN-tanh"     "tcn/model_bb_tcn-45-s-16"
run_one "LSTM"         "lstm/model_bb_lstm-32"
run_one "S4"           "s4/model_bb_s4-b4-s4-c16"

echo ""
echo "========== multirun sanity =========="
uv run nablafx -m \
  "${COMMON[@]}" \
  model=tcn/model_bb_tcn-45-s-16,lstm/model_bb_lstm-32

echo ""
echo "[smoke] OK — all Hydra cases passed"

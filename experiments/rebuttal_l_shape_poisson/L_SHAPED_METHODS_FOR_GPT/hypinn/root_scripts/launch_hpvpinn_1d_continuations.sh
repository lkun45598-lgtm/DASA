#!/usr/bin/env bash
set -euo pipefail

PYTHON=/home/whn/miniconda3/envs/nan717/bin/python
RUNNER=/home/whn/hpy_pinns/hpvpinn_experiments.py
ROOT=/home/whn/hpy_pinns
SEED=${1:-1234}
GPU=${2:-7}
RUN_DIR="$ROOT/runs/table1_optimized_seed_${SEED}"
RESULT_ROOT="$ROOT/results/table1_hpvpinn_optimized"

mkdir -p "$RUN_DIR"

launch() {
  local problem="$1"
  local source="$ROOT/results/table1_hpvpinn/$problem/seed_$SEED/checkpoint_latest.pt"
  local output="$RESULT_ROOT/$problem/adam_continuation_lr1e-5_seed_$SEED"
  local log="$RUN_DIR/${problem}_continuation.log"
  local session="hpv_cont_${SEED}_${problem}"
  if tmux has-session -t "$session" 2>/dev/null; then
    return
  fi
  local command="cd $ROOT && exec env CUDA_VISIBLE_DEVICES=$GPU $PYTHON -u $RUNNER \
    --problem $problem --seed $SEED --device cuda --dtype float64 \
    --output-dir $output --iterations 20001 --variational-form strong \
    --mesh-variant baseline --init-checkpoint $source --learning-rate 1e-5 --lr-gamma 1.0 \
    > $log 2>&1"
  tmux new-session -d -s "$session" "$command"
}

launch poisson_1d_boundary
launch poisson_1d_steep

tmux list-sessions -F '#{session_name}\t#{pane_pid}' | rg "^hpv_cont_${SEED}_"

#!/usr/bin/env bash
set -euo pipefail

PYTHON=/home/whn/miniconda3/envs/nan717/bin/python
RUNNER=/home/whn/hpy_pinns/hpvpinn_experiments.py
ROOT=/home/whn/hpy_pinns
SEED=${1:-1234}
RESULT_ROOT="$ROOT/results/table1_hpvpinn_kappa001/advection_diffusion_1d"
RUN_DIR="$ROOT/runs/table1_kappa001_seed_${SEED}"

mkdir -p "$RUN_DIR"

launch() {
  local label="$1" gpu="$2"; shift 2
  local output="$RESULT_ROOT/${label}_seed_$SEED"
  local log="$RUN_DIR/${label}.log"
  local session="hpv_k001_${SEED}_${label}"
  if tmux has-session -t "$session" 2>/dev/null; then
    return
  fi
  local command="cd $ROOT && exec env CUDA_VISIBLE_DEVICES=$gpu $PYTHON -u $RUNNER \
    --problem advection_diffusion_1d --seed $SEED --device cuda --dtype float64 \
    --output-dir $output --iterations 21001 --variational-form p1 \
    --boundary-weight 10 --initial-weight 10 --constraint-aggregation combined \
    --ade-kappa 0.01 --lr-gamma 1.0 $* > $log 2>&1"
  tmux new-session -d -s "$session" "$command"
}

# h-refinement, p-refinement, and Adam stability are separate reproducible variants.
launch h8_p5_q10_lr1e-3 2 --mesh-variant refined --n-test 5 --n-quad 10 --learning-rate 1e-3
launch h4_p8_q16_lr1e-4 3 --mesh-variant official --n-test 8 --n-quad 16 --learning-rate 1e-4
launch h4_p5_q10_lr1e-4 5 --mesh-variant official --n-test 5 --n-quad 10 --learning-rate 1e-4

tmux list-sessions -F '#{session_name}\t#{pane_pid}' | rg "^hpv_k001_${SEED}_" | sort

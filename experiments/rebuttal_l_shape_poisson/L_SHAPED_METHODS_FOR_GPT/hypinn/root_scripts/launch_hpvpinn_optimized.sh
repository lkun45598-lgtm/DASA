#!/usr/bin/env bash
set -euo pipefail

# Faithful hp-VPINN variants for the five DASA reference-anchor cases.
# Every command is deterministic and writes to a new result tree.
PYTHON=/home/whn/miniconda3/envs/nan717/bin/python
RUNNER=/home/whn/hpy_pinns/hpvpinn_experiments.py
ROOT=/home/whn/hpy_pinns
SEED=${1:-1234}
RUN_DIR="$ROOT/runs/table1_optimized_seed_${SEED}"
RESULT_ROOT="$ROOT/results/table1_hpvpinn_optimized"

mkdir -p "$RUN_DIR"
: > "$RUN_DIR/pids.tsv"

run_one() {
  local name="$1" gpu="$2" output="$3"; shift 3
  local log="$RUN_DIR/$name.log"
  local session="hpv_opt_${SEED}_${name}"
  if tmux has-session -t "$session" 2>/dev/null; then
    local pane_pid
    pane_pid=$(tmux display-message -p -t "$session" '#{pane_pid}')
    printf '%s\t%s\t%s\t%s\t%s\n' "$name" "$gpu" "$pane_pid" "$session" "$log" >> "$RUN_DIR/pids.tsv"
    return
  fi
  local command
  command="cd $ROOT && exec env CUDA_VISIBLE_DEVICES=$gpu $PYTHON -u $RUNNER --problem $name --seed $SEED --device cuda --dtype float64 --output-dir $output $* > $log 2>&1"
  tmux new-session -d -s "$session" "$command"
  local pane_pid
  pane_pid=$(tmux display-message -p -t "$session" '#{pane_pid}')
  printf '%s\t%s\t%s\t%s\t%s\n' "$name" "$gpu" "$pane_pid" "$session" "$log" >> "$RUN_DIR/pids.tsv"
}

run_one poisson_1d_boundary 1 "$RESULT_ROOT/poisson_1d_boundary/constant_adam_seed_$SEED" \
  --iterations 40001 --variational-form strong --mesh-variant baseline --lr-gamma 1.0
run_one poisson_1d_steep 2 "$RESULT_ROOT/poisson_1d_steep/constant_adam_seed_$SEED" \
  --iterations 40001 --variational-form strong --mesh-variant baseline --lr-gamma 1.0
run_one poisson_2d_steep 3 "$RESULT_ROOT/poisson_2d_steep/p1_8x8_seed_$SEED" \
  --iterations 21001 --variational-form p1 --mesh-variant official --boundary-weight 10 --lr-gamma 1.0
run_one advection_diffusion_1d 4 "$RESULT_ROOT/advection_diffusion_1d/p1_4x4_seed_$SEED" \
  --iterations 21001 --variational-form p1 --mesh-variant official --boundary-weight 10 --initial-weight 10 \
  --constraint-aggregation combined --lr-gamma 1.0
run_one poisson_lshape 5 "$RESULT_ROOT/poisson_lshape/strong_35elem_seed_$SEED" \
  --iterations 28001 --variational-form strong --mesh-variant official --boundary-weight 10 --lr-gamma 1.0

cat "$RUN_DIR/pids.tsv"

#!/usr/bin/env bash
set -euo pipefail

# Re-run the historical baseline configurations after the tensor-product
# quadrature ordering fix in hpvpinn_experiments.py.
PYTHON=/home/whn/miniconda3/envs/nan717/bin/python
RUNNER=/home/whn/hpy_pinns/hpvpinn_experiments.py
ROOT=/home/whn/hpy_pinns
SEED=${1:-1234}
RUN_DIR="$ROOT/runs/table1_corrected_seed_${SEED}"
RESULT_ROOT="$ROOT/results/table1_hpvpinn_corrected"

mkdir -p "$RUN_DIR"
: > "$RUN_DIR/pids.tsv"

run_one() {
  local problem="$1" gpu="$2" iterations="$3"
  local output="$RESULT_ROOT/$problem/seed_$SEED"
  local log="$RUN_DIR/$problem.log"
  local session="hpv_corrected_${SEED}_${problem}"
  if tmux has-session -t "$session" 2>/dev/null; then
    local pane_pid
    pane_pid=$(tmux display-message -p -t "$session" '#{pane_pid}')
    printf '%s\t%s\t%s\t%s\t%s\n' "$problem" "$gpu" "$pane_pid" "$session" "$log" >> "$RUN_DIR/pids.tsv"
    return
  fi
  local command="cd $ROOT && exec env CUDA_VISIBLE_DEVICES=$gpu $PYTHON -u $RUNNER \
    --problem $problem --seed $SEED --device cuda --dtype float64 \
    --output-dir $output --iterations $iterations --variational-form strong \
    --mesh-variant baseline --lr-gamma 0.9 --resume > $log 2>&1"
  tmux new-session -d -s "$session" "$command"
  local pane_pid
  pane_pid=$(tmux display-message -p -t "$session" '#{pane_pid}')
  printf '%s\t%s\t%s\t%s\t%s\n' "$problem" "$gpu" "$pane_pid" "$session" "$log" >> "$RUN_DIR/pids.tsv"
}

# Two independent jobs per free 24 GB card; these baseline runs use modest
# memory, while the existing workloads on GPUs 0, 4, and 7 are left alone.
run_one poisson_2d_peak 2 21001
run_one helmholtz_1d 2 21001
run_one helmholtz_2d 3 21001
run_one cavity_re100 3 25001
run_one burgers_1d 5 40001
run_one allen_cahn_1d 5 60001
run_one diffusion_reaction_1d 6 21001
run_one advection_discontinuous_1d 6 26001

# These two previously selected 2D candidates predate the ordering fix.
# Keep their published hp-VPINN settings, but write corrected results separately.
run_one_p1_steep() {
  local problem=poisson_2d_steep gpu=1 iterations=21001
  local output="$RESULT_ROOT/$problem/p1_8x8_seed_$SEED"
  local log="$RUN_DIR/$problem.log"
  local session="hpv_corrected_${SEED}_${problem}"
  if tmux has-session -t "$session" 2>/dev/null; then
    return
  fi
  tmux new-session -d -s "$session" "cd $ROOT && exec env CUDA_VISIBLE_DEVICES=$gpu $PYTHON -u $RUNNER \
    --problem $problem --seed $SEED --device cuda --dtype float64 --output-dir $output \
    --iterations $iterations --variational-form p1 --mesh-variant official --boundary-weight 10 \
    --lr-gamma 1.0 --resume > $log 2>&1"
  local pane_pid
  pane_pid=$(tmux display-message -p -t "$session" '#{pane_pid}')
  printf '%s\t%s\t%s\t%s\t%s\n' "$problem" "$gpu" "$pane_pid" "$session" "$log" >> "$RUN_DIR/pids.tsv"
}

run_one_lshape() {
  local problem=poisson_lshape gpu=1 iterations=28001
  local output="$RESULT_ROOT/$problem/strong_35elem_seed_$SEED"
  local log="$RUN_DIR/$problem.log"
  local session="hpv_corrected_${SEED}_${problem}"
  if tmux has-session -t "$session" 2>/dev/null; then
    return
  fi
  tmux new-session -d -s "$session" "cd $ROOT && exec env CUDA_VISIBLE_DEVICES=$gpu $PYTHON -u $RUNNER \
    --problem $problem --seed $SEED --device cuda --dtype float64 --output-dir $output \
    --iterations $iterations --variational-form strong --mesh-variant official --boundary-weight 10 \
    --lr-gamma 0.9 --resume > $log 2>&1"
  local pane_pid
  pane_pid=$(tmux display-message -p -t "$session" '#{pane_pid}')
  printf '%s\t%s\t%s\t%s\t%s\n' "$problem" "$gpu" "$pane_pid" "$session" "$log" >> "$RUN_DIR/pids.tsv"
}

run_one_p1_steep
run_one_lshape

cat "$RUN_DIR/pids.tsv"

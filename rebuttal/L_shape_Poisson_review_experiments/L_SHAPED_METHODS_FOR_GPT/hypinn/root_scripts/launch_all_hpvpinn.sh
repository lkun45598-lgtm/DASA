#!/usr/bin/env bash
set -euo pipefail

PYTHON=/home/whn/miniconda3/envs/nan717/bin/python
RUNNER=/home/whn/hpy_pinns/hpvpinn_experiments.py
ROOT=/home/whn/hpy_pinns
SEED=${1:-1234}
RUN_DIR="$ROOT/runs/table1_seed_${SEED}"
RESULT_ROOT="$ROOT/results/table1_hpvpinn"

mkdir -p "$RUN_DIR"
: > "$RUN_DIR/pids.tsv"

problems=(
  poisson_1d_boundary
  poisson_1d_steep
  helmholtz_1d
  poisson_2d_peak
  poisson_2d_steep
  helmholtz_2d
  cavity_re100
  burgers_1d
  allen_cahn_1d
  diffusion_reaction_1d
  advection_diffusion_1d
  poisson_lshape
  advection_discontinuous_1d
)

gpus=(0 0 1 2 3 4 5 4 6 2 3 7 1)

for index in "${!problems[@]}"; do
  problem=${problems[$index]}
  gpu=${gpus[$index]}
  output="$RESULT_ROOT/$problem/seed_$SEED"
  log="$RUN_DIR/$problem.log"

  session="hpv_${SEED}_${problem}"
  if tmux has-session -t "$session" 2>/dev/null; then
    pane_pid=$(tmux display-message -p -t "$session" '#{pane_pid}')
    printf '%s\t%s\t%s\t%s\t%s\n' "$problem" "$gpu" "$pane_pid" "$session" "$log" >> "$RUN_DIR/pids.tsv"
    continue
  fi

  command="cd $ROOT && exec env CUDA_VISIBLE_DEVICES=$gpu $PYTHON -u $RUNNER --problem $problem --seed $SEED --device cuda --dtype float64 --resume --output-dir $output > $log 2>&1"
  tmux new-session -d -s "$session" "$command"
  pane_pid=$(tmux display-message -p -t "$session" '#{pane_pid}')
  printf '%s\t%s\t%s\t%s\t%s\n' "$problem" "$gpu" "$pane_pid" "$session" "$log" >> "$RUN_DIR/pids.tsv"
done

cat "$RUN_DIR/pids.tsv"

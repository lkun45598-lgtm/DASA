#!/usr/bin/env bash
set -euo pipefail

PYTHON=/home/whn/miniconda3/envs/nan717/bin/python
RUNNER=/home/whn/hpy_pinns/hpvpinn_experiments.py
ROOT=/home/whn/hpy_pinns
SEED=${1:-1234}
GPU=${2:-3}
PROBLEM=poisson_2d_steep
SOURCE="$ROOT/results/table1_hpvpinn_optimized/$PROBLEM/p1_8x8_100k_seed_$SEED/checkpoint_latest.pt"
OUTPUT="$ROOT/results/table1_hpvpinn_optimized/$PROBLEM/p1_p8_q16_continuation_seed_$SEED"
LOG="$ROOT/runs/table1_optimized_seed_${SEED}/${PROBLEM}_p8_q16.log"
SESSION="hpv_2d_prefine_${SEED}_${PROBLEM}"

if [[ ! -f "$SOURCE" || ! -f "$(dirname "$SOURCE")/metrics.json" ]]; then
  printf 'source 100k run has not completed: %s\n' "$SOURCE" >&2
  exit 1
fi

if ! tmux has-session -t "$SESSION" 2>/dev/null; then
  COMMAND="cd $ROOT && exec env CUDA_VISIBLE_DEVICES=$GPU $PYTHON -u $RUNNER \
    --problem $PROBLEM --seed $SEED --device cuda --dtype float64 \
    --output-dir $OUTPUT --iterations 30001 --variational-form p1 --mesh-variant official \
    --n-test 8 --n-quad 16 --boundary-weight 10 --init-checkpoint $SOURCE \
    --learning-rate 1e-4 --lr-gamma 1.0 > $LOG 2>&1"
  tmux new-session -d -s "$SESSION" "$COMMAND"
fi

tmux display-message -p -t "$SESSION" '#{session_name}\t#{pane_pid}'

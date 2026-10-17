#!/usr/bin/env bash
set -euo pipefail

PYTHON=/home/whn/miniconda3/envs/nan717/bin/python
RUNNER=/home/whn/hpy_pinns/hpvpinn_experiments.py
ROOT=/home/whn/hpy_pinns
SEED=${1:-1234}
GPU=${2:-4}
PROBLEM=advection_diffusion_1d
SOURCE="$ROOT/results/table1_hpvpinn_optimized/$PROBLEM/p1_4x4_seed_$SEED/checkpoint_latest.pt"
OUTPUT="$ROOT/results/table1_hpvpinn_optimized/$PROBLEM/p1_4x4_100k_lr1e-4_seed_$SEED"
LOG="$ROOT/runs/table1_optimized_seed_${SEED}/${PROBLEM}_100k_lr1e-4.log"
SESSION="hpv_ade_low_lr_${SEED}"

if [[ ! -f "$SOURCE" || ! -f "$(dirname "$SOURCE")/metrics.json" ]]; then
  printf 'source 21k run has not completed: %s\n' "$SOURCE" >&2
  exit 1
fi

if ! tmux has-session -t "$SESSION" 2>/dev/null; then
  COMMAND="cd $ROOT && exec env CUDA_VISIBLE_DEVICES=$GPU $PYTHON -u $RUNNER \
    --problem $PROBLEM --seed $SEED --device cuda --dtype float64 \
    --output-dir $OUTPUT --iterations 79000 --variational-form p1 --mesh-variant official \
    --boundary-weight 10 --initial-weight 10 --constraint-aggregation combined \
    --init-checkpoint $SOURCE --learning-rate 1e-4 --lr-gamma 1.0 > $LOG 2>&1"
  tmux new-session -d -s "$SESSION" "$COMMAND"
fi

tmux display-message -p -t "$SESSION" '#{session_name}\t#{pane_pid}'

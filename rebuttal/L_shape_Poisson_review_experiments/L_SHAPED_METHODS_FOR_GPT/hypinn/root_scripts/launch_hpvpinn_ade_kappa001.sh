#!/usr/bin/env bash
set -euo pipefail

PYTHON=/home/whn/miniconda3/envs/nan717/bin/python
RUNNER=/home/whn/hpy_pinns/hpvpinn_experiments.py
ROOT=/home/whn/hpy_pinns
SEED=${1:-1234}
GPU=${2:-1}
OUTPUT="$ROOT/results/table1_hpvpinn_kappa001/advection_diffusion_1d/p1_4x4_seed_$SEED"
LOG="$ROOT/runs/table1_kappa001_seed_${SEED}/advection_diffusion_1d.log"
SESSION="hpv_k001_${SEED}_ade"

mkdir -p "$(dirname "$LOG")"
if ! tmux has-session -t "$SESSION" 2>/dev/null; then
  COMMAND="cd $ROOT && exec env CUDA_VISIBLE_DEVICES=$GPU $PYTHON -u $RUNNER \
    --problem advection_diffusion_1d --seed $SEED --device cuda --dtype float64 \
    --output-dir $OUTPUT --iterations 21001 --variational-form p1 --mesh-variant official \
    --boundary-weight 10 --initial-weight 10 --constraint-aggregation combined \
    --learning-rate 1e-3 --lr-gamma 1.0 --ade-kappa 0.01 > $LOG 2>&1"
  tmux new-session -d -s "$SESSION" "$COMMAND"
fi

tmux display-message -p -t "$SESSION" '#{session_name}\t#{pane_pid}'

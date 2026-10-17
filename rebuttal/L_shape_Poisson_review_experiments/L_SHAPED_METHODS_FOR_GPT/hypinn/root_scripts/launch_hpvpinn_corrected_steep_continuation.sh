#!/usr/bin/env bash
set -euo pipefail

# Continue only from corrected checkpoints. This mirrors the previously used
# 21,001 -> 100,001 -> p=8/Q=16 continuation chain for 2D Poisson steep.
PYTHON=/home/whn/miniconda3/envs/nan717/bin/python
RUNNER=/home/whn/hpy_pinns/hpvpinn_experiments.py
ROOT=/home/whn/hpy_pinns
SEED=${1:-1234}
RUN_DIR="$ROOT/runs/table1_corrected_seed_${SEED}"
RESULT_ROOT="$ROOT/results/table1_hpvpinn_corrected/poisson_2d_steep"
BASE="$RESULT_ROOT/p1_8x8_seed_$SEED/checkpoint_latest.pt"
STAGE2="$RESULT_ROOT/p1_8x8_100k_seed_$SEED/checkpoint_latest.pt"

if [[ ! -f "$BASE" || ! -f "$RESULT_ROOT/p1_8x8_seed_$SEED/metrics.json" ]]; then
  echo "waiting: corrected initial 2D Poisson steep stage has not completed"
  exit 2
fi

session="hpv_corrected_${SEED}_poisson_2d_steep_100k"
if ! tmux has-session -t "$session" 2>/dev/null && [[ ! -f "$RESULT_ROOT/p1_8x8_100k_seed_$SEED/metrics.json" ]]; then
  tmux new-session -d -s "$session" "cd $ROOT && exec env CUDA_VISIBLE_DEVICES=2 $PYTHON -u $RUNNER \
    --problem poisson_2d_steep --seed $SEED --device cuda --dtype float64 \
    --output-dir $RESULT_ROOT/p1_8x8_100k_seed_$SEED --iterations 79000 \
    --init-checkpoint $BASE --variational-form p1 --mesh-variant official --boundary-weight 10 \
    --lr-gamma 1.0 > $RUN_DIR/poisson_2d_steep_100k.log 2>&1"
fi

if [[ ! -f "$STAGE2" || ! -f "$RESULT_ROOT/p1_8x8_100k_seed_$SEED/metrics.json" ]]; then
  echo "launched_or_running: corrected p1 100k stage"
  exit 0
fi

session="hpv_corrected_${SEED}_poisson_2d_steep_p8q16"
if ! tmux has-session -t "$session" 2>/dev/null && [[ ! -f "$RESULT_ROOT/p1_p8_q16_continuation_seed_$SEED/metrics.json" ]]; then
  tmux new-session -d -s "$session" "cd $ROOT && exec env CUDA_VISIBLE_DEVICES=3 $PYTHON -u $RUNNER \
    --problem poisson_2d_steep --seed $SEED --device cuda --dtype float64 \
    --output-dir $RESULT_ROOT/p1_p8_q16_continuation_seed_$SEED --iterations 30001 \
    --init-checkpoint $STAGE2 --variational-form p1 --mesh-variant official --boundary-weight 10 \
    --n-test 8 --n-quad 16 --learning-rate 1e-4 --lr-gamma 1.0 > $RUN_DIR/poisson_2d_steep_p8_q16.log 2>&1"
fi

echo "launched_or_running: corrected p=8/Q=16 stage"

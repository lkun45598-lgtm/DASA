#!/usr/bin/env bash
set -euo pipefail

PYTHON=/home/whn/miniconda3/envs/nan717/bin/python
RUNNER=/home/whn/hpy_pinns/hpvpinn_experiments.py
ROOT=/home/whn/hpy_pinns
SEED=${1:-1234}
RUN_DIR="$ROOT/runs/table1_optimized_seed_${SEED}"
RESULT_ROOT="$ROOT/results/table1_hpvpinn_optimized"

mkdir -p "$RUN_DIR"

launch() {
  local label="$1" problem="$2" gpu="$3" source="$4" output="$5" steps="$6"; shift 6
  local log="$RUN_DIR/${label}.log"
  local session="hpv_100k_${SEED}_${label}"
  if [[ ! -f "$source" || ! -f "$(dirname "$source")/metrics.json" ]]; then
    printf 'waiting for source checkpoint: %s\n' "$source"
    return
  fi
  if tmux has-session -t "$session" 2>/dev/null; then
    return
  fi
  local command="cd $ROOT && exec env CUDA_VISIBLE_DEVICES=$gpu $PYTHON -u $RUNNER \
    --problem $problem --seed $SEED --device cuda --dtype float64 \
    --output-dir $output --iterations $steps --init-checkpoint $source \
    --lr-gamma 1.0 $* > $log 2>&1"
  tmux new-session -d -s "$session" "$command"
}

launch poisson_2d_steep_100k poisson_2d_steep 3 \
  "$RESULT_ROOT/poisson_2d_steep/p1_8x8_seed_$SEED/checkpoint_latest.pt" \
  "$RESULT_ROOT/poisson_2d_steep/p1_8x8_100k_seed_$SEED" 79000 \
  --variational-form p1 --mesh-variant official --boundary-weight 10

launch advection_diffusion_1d_100k advection_diffusion_1d 4 \
  "$RESULT_ROOT/advection_diffusion_1d/p1_4x4_seed_$SEED/checkpoint_latest.pt" \
  "$RESULT_ROOT/advection_diffusion_1d/p1_4x4_100k_seed_$SEED" 79000 \
  --variational-form p1 --mesh-variant official --boundary-weight 10 --initial-weight 10 \
  --constraint-aggregation combined

tmux list-sessions -F '#{session_name}\t#{pane_pid}' | rg "^hpv_100k_${SEED}_" || true

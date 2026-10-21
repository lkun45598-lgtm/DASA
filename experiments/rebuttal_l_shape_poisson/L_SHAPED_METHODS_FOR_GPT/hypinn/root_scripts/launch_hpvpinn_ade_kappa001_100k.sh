#!/usr/bin/env bash
set -euo pipefail

PYTHON=/home/whn/miniconda3/envs/nan717/bin/python
RUNNER=/home/whn/hpy_pinns/hpvpinn_experiments.py
ROOT=/home/whn/hpy_pinns
SEED=${1:-1234}
RESULT_ROOT="$ROOT/results/table1_hpvpinn_kappa001/advection_diffusion_1d"
RUN_DIR="$ROOT/runs/table1_kappa001_seed_${SEED}"

launch() {
  local label="$1" gpu="$2" source_variant="$3" lr="$4"; shift 4
  local source="$RESULT_ROOT/${source_variant}_seed_$SEED/checkpoint_latest.pt"
  local output="$RESULT_ROOT/${label}_seed_$SEED"
  local log="$RUN_DIR/${label}.log"
  local session="hpv_k001_100k_${SEED}_${label}"
  if [[ ! -f "$source" || ! -f "$(dirname "$source")/metrics.json" ]]; then
    printf 'source run has not completed: %s\n' "$source" >&2
    exit 1
  fi
  if tmux has-session -t "$session" 2>/dev/null; then
    return
  fi
  local command="cd $ROOT && exec env CUDA_VISIBLE_DEVICES=$gpu $PYTHON -u $RUNNER \
    --problem advection_diffusion_1d --seed $SEED --device cuda --dtype float64 \
    --output-dir $output --iterations 79000 --variational-form p1 \
    --boundary-weight 10 --initial-weight 10 --constraint-aggregation combined \
    --ade-kappa 0.01 --init-checkpoint $source --learning-rate $lr --lr-gamma 1.0 \
    $* > $log 2>&1"
  tmux new-session -d -s "$session" "$command"
}

launch h4_p5_q10_100k_lr1e-4 1 p1_4x4 1e-4 --mesh-variant official --n-test 5 --n-quad 10
launch h8_p5_q10_100k_lr1e-4 2 h8_p5_q10_lr1e-3 1e-4 --mesh-variant refined --n-test 5 --n-quad 10
launch h4_p8_q16_100k_lr1e-5 3 h4_p8_q16_lr1e-4 1e-5 --mesh-variant official --n-test 8 --n-quad 16
launch h4_p5_q10_100k_lr1e-5 5 h4_p5_q10_lr1e-4 1e-5 --mesh-variant official --n-test 5 --n-quad 10

tmux list-sessions -F '#{session_name}\t#{pane_pid}' | rg "^hpv_k001_100k_${SEED}_" | sort

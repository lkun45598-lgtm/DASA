#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUNNER="${RUNNER:-$ROOT/L_SHAPED_METHODS_FOR_GPT/refactored_dasa_pinn/src/lshape_experiment.py}"
DATA="${DATA:-$ROOT/data/Lpoission_more.mat}"
PYTHON="${PYTHON:-/home/lz/miniconda3/envs/pytorch/bin/python}"
DEVICE="${DEVICE:-cuda:0}"
SEEDS="${SEEDS:-0 1 2}"
RESULTS="${RESULTS:-$ROOT/results_stage_length_sensitivity}"
CONFIG_DIR="$ROOT/configs/stage_length"
CONFIGS="${CONFIGS:-$CONFIG_DIR/*.json}"

run_one() {
  local config="$1" seed="$2" output_dir="$RESULTS/$(basename "${config%.json}")/seed_$seed"
  local metrics="$output_dir/metrics.json"
  if [[ -f "$metrics" ]]; then
    echo "[skip] $metrics"
    return 0
  fi
  mkdir -p "$output_dir"
  echo "[$(date -Is)] $(basename "$config") seed $seed device $DEVICE" | tee -a "$output_dir/train.log"
  PYTHONUNBUFFERED=1 \
    "$PYTHON" "$RUNNER" --config "$config" --seed "$seed" --device "$DEVICE" \
    --data "$DATA" --output-root "$RESULTS" 2>&1 | tee -a "$output_dir/train.log"
}

if [[ "${1:-}" == "--aggregate" ]]; then
  "$PYTHON" "$ROOT/src/aggregate_stage_length.py" --results "$RESULTS" --out "$RESULTS/summaries"
  exit 0
fi

for config in $CONFIGS; do
  for seed in $SEEDS; do
    run_one "$config" "$seed"
  done
done

"$PYTHON" "$ROOT/src/aggregate_stage_length.py" --results "$RESULTS" --out "$RESULTS/summaries"

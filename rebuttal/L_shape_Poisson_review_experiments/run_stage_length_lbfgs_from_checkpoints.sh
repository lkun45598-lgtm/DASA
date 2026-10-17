#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUNNER="$ROOT/L_SHAPED_METHODS_FOR_GPT/refactored_dasa_pinn/src/lshape_experiment.py"
SOURCE="${SOURCE:-$ROOT/results_stage_length_sensitivity_controlled_lr}"
RESULTS="${RESULTS:-$ROOT/results_stage_length_sensitivity_controlled_lr_lbfgs}"
PYTHON="${PYTHON:-/home/lz/miniconda3/envs/pytorch/bin/python}"
DEVICE="${DEVICE:-cuda:0}"
CONFIGS="${CONFIGS:-$ROOT/configs/stage_length/*.json}"
SEEDS="${SEEDS:-0 1 2}"

for config in $CONFIGS; do
  experiment="$(basename "${config%.json}")"
  for seed in $SEEDS; do
    source_dir="$SOURCE/$experiment/seed_$seed"
    output_dir="$RESULTS/$experiment/seed_$seed"
    if [[ -f "$output_dir/metrics.json" ]]; then
      echo "[skip] $output_dir/metrics.json"
      continue
    fi
    mkdir -p "$output_dir"
    PYTHONUNBUFFERED=1 "$PYTHON" "$RUNNER" \
      --config "$config" --seed "$seed" --device "$DEVICE" \
      --data "$ROOT/data/Lpoission_more.mat" --output-root "$RESULTS" \
      --lbfgs-only --checkpoint "$source_dir/final_model.pt" \
      --base-metrics "$source_dir/metrics.json" \
      --lbfgs-outer-steps 1000 --lbfgs-max-iter 100 \
      2>&1 | tee -a "$output_dir/train.log"
  done
done

"$PYTHON" "$ROOT/src/aggregate_stage_length.py" \
  --results "$RESULTS" --out "$RESULTS/summaries" --evaluation-label final

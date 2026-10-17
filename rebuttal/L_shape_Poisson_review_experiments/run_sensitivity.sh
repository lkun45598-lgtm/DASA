#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEVICE="${DEVICE:-cuda:0}"
SEEDS="${SEEDS:-0 1 2}"
PROFILE="${PROFILE:-paper}"
RESULTS="${RESULTS:-$ROOT/results_sensitivity_${PROFILE}}"

case "$PROFILE" in
  paper)
    # 5000 warm-up + 2000 subnet pretraining + 10000 adversarial steps.
    # Evaluate immediately after the adversarial stage; no finetune/L-BFGS.
    EXTRA=()
    ;;
  quick)
    # Screening only; do not report these numbers in the rebuttal.
    EXTRA=(--warmup-steps 500 --subnet-pretrain-steps 200 --main-steps 1000 --finetune-steps 0 --lbfgs-outer-steps 0 --log-every 250)
    ;;
  smoke)
    EXTRA=(--warmup-steps 2 --subnet-pretrain-steps 2 --main-steps 2 --finetune-steps 0 --lbfgs-outer-steps 0 --log-every 1 --n-interior-candidates 64 --n-boundary-per-segment 8)
    ;;
  *)
    echo "Unknown PROFILE=$PROFILE (use paper, quick, or smoke)" >&2
    exit 2
    ;;
esac

CONFIG_GLOB="${CONFIG_GLOB:-$ROOT/configs/sensitivity/*.json}"
for cfg in $CONFIG_GLOB; do
  for seed in $SEEDS; do
    python "$ROOT/src/lshape_experiment.py" \
      --config "$cfg" \
      --seed "$seed" \
      --device "$DEVICE" \
      --output-root "$RESULTS" \
      "${EXTRA[@]}"
  done
done

python "$ROOT/src/aggregate_results.py" \
  --results "$RESULTS" \
  --out "$RESULTS/summaries"

echo "Sensitivity table: $RESULTS/summaries/sensitivity_summary.md"

#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEVICE="${DEVICE:-cuda:0}"
SEEDS="${SEEDS:-0 1 2}"
PROFILE="${PROFILE:-adam}"
RESULTS="${RESULTS:-$ROOT/results_cost_${PROFILE}}"

case "$PROFILE" in
  adam)
    # Safe first pass: all Adam stages, no L-BFGS.
    EXTRA=(--lbfgs-outer-steps 0)
    ;;
  full)
    # Exact supplied notebook setting: 1000 outer L-BFGS calls, each max_iter=100.
    # This can dominate total runtime. Confirm that this was truly used in the paper.
    EXTRA=()
    ;;
  smoke)
    EXTRA=(--warmup-steps 2 --subnet-pretrain-steps 2 --main-steps 2 --finetune-steps 2 --lbfgs-outer-steps 0 --log-every 1 --n-interior-candidates 64 --n-boundary-per-segment 8)
    ;;
  *)
    echo "Unknown PROFILE=$PROFILE (use adam, full, or smoke)" >&2
    exit 2
    ;;
esac

for cfg in "$ROOT/configs/cost/cost_pinn.json" "$ROOT/configs/cost/cost_dasa.json"; do
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

echo "Cost tables: $RESULTS/summaries/cost_summary.md"

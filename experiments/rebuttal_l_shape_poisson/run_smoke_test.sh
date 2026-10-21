#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEVICE="${DEVICE:-cpu}" SEEDS="0" PROFILE="smoke" RESULTS="$ROOT/results_smoke_cost" bash "$ROOT/run_cost_comparison.sh"
DEVICE="${DEVICE:-cpu}" SEEDS="0" PROFILE="smoke" CONFIG_GLOB="$ROOT/configs/sensitivity/sens_baseline.json" RESULTS="$ROOT/results_smoke_sensitivity" bash "$ROOT/run_sensitivity.sh"

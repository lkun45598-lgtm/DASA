#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SUITE_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PROJECT_ROOT="$(cd "$SUITE_ROOT/../.." && pwd)"
PYTHON="/home/lz/miniconda3/envs/pytorch/bin/python"
RUN_ROOT="${1:-$SUITE_ROOT/runs/smoke_cpu}"

if [[ -e "$RUN_ROOT" ]]; then
    echo "smoke output already exists: $RUN_ROOT" >&2
    exit 1
fi
mkdir -p "$RUN_ROOT"

cases=(
    advection_diffusion
    allen_cahn
    burgers
    cavity
    diffusion_reaction
    discontinuous_advection
    helmholtz_1d
    helmholtz_2d
    lshape_poisson
    poisson_1d_boundary
    poisson_1d_steep
    poisson_2d_peak
    poisson_2d_steep
)

cd "$PROJECT_ROOT"
for case_id in "${cases[@]}"; do
    output="$RUN_ROOT/seed_0/$case_id"
    mkdir -p "$output"
    DDE_BACKEND=pytorch MPLCONFIGDIR=/tmp/matplotlib-deepxde CUDA_VISIBLE_DEVICES='' \
        "$PYTHON" -m benchmarks.deepxde_pinn.scripts.run_case \
        --case "$case_id" --seed 0 --output "$output" \
        --adam-steps 1 --lbfgs-steps 0 \
        --residual-points 8 --boundary-points 8 --initial-points 8 \
        >"$output/train.log" 2>&1
done

DDE_BACKEND=pytorch MPLCONFIGDIR=/tmp/matplotlib-deepxde \
    "$PYTHON" -m benchmarks.deepxde_pinn.scripts.aggregate_results --root "$RUN_ROOT"

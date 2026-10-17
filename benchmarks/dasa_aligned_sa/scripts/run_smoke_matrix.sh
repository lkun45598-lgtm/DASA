#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SUITE_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PROJECT_ROOT="$(cd "$SUITE_ROOT/../.." && pwd)"
PYTHON="$PROJECT_ROOT/.conda-env/bin/python"
RUN_ROOT="${1:-$SUITE_ROOT/runs/smoke}"

if [[ -e "$RUN_ROOT" ]]; then
    echo "smoke output already exists: $RUN_ROOT" >&2
    exit 1
fi
for dataset in "$PROJECT_ROOT/Burgers/burgers_shock.mat" "$PROJECT_ROOT/Allen-Cahn/AC.mat" "$SUITE_ROOT/data/cavity_reference_u.csv"; do
    if [[ ! -f "$dataset" ]]; then
        echo "missing original SA-PINN dataset: $dataset" >&2
        exit 1
    fi
done
for reference in lshape_poisson; do
    if [[ ! -f "$SUITE_ROOT/references/$reference.npz" || ! -f "$SUITE_ROOT/references/$reference.json" ]]; then
        echo "missing frozen reference: $reference" >&2
        exit 1
    fi
done

mkdir -p "$RUN_ROOT"
PIDS=()

smoke_case() {
    local case_id="$1"
    local gpu_id="$2"
    local output="$RUN_ROOT/$case_id"
    mkdir -p "$output"
    (
        cd "$PROJECT_ROOT"
        CUDA_VISIBLE_DEVICES="$gpu_id" TF_FORCE_GPU_ALLOW_GROWTH=true \
            "$PYTHON" -m benchmarks.dasa_aligned_sa.scripts.run_case \
            --case "$case_id" --seed 0 --output "$output" \
            --adam-steps 2 --lbfgs-steps 1 >"$output/train.log" 2>&1
    ) &
    PIDS+=("$!")
}

smoke_case allen_cahn 0
smoke_case diffusion_reaction 1
smoke_case advection_diffusion 2
smoke_case cavity 3
smoke_case poisson_1d_boundary 3
smoke_case lshape_poisson 4
smoke_case poisson_1d_steep 4
smoke_case helmholtz_2d 5
smoke_case helmholtz_1d 5
smoke_case burgers 6
smoke_case poisson_2d_peak 6
smoke_case discontinuous_advection 7
smoke_case poisson_2d_steep 7

status=0
for pid in "${PIDS[@]}"; do
    if ! wait "$pid"; then
        status=1
    fi
done

cd "$PROJECT_ROOT"
"$PYTHON" -m benchmarks.dasa_aligned_sa.scripts.aggregate_results --root "$RUN_ROOT"
exit "$status"

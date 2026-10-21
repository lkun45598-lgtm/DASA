#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
CASE="${1:-all}"

export CUDA_VISIBLE_DEVICES=7
export DDE_BACKEND=pytorch
export PYTHONPATH="${ROOT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"

ADAM_ITERATIONS="${ADAM_ITERATIONS:-40000}"
LBFGS_ITERATIONS="${LBFGS_ITERATIONS:-0}"
DISPLAY_EVERY="${DISPLAY_EVERY:-1000}"
SEEDS="${SEEDS:-2027 2028 2029 2030 2031 2032 2033}"
OUTPUT_DIR="${OUTPUT_DIR:-runs/multiseed_modified_mlp_corrected/poisson_1d}"

run_case() {
    local case_name="$1"
    local seed="$2"
    local seed_output_dir="${OUTPUT_DIR}/seed_${seed}"
    echo "Running 1D Poisson case=${case_name}, seed=${seed}, output=${seed_output_dir}"
    python -m pinn_baselines.poisson_1d \
        --case "${case_name}" \
        --seed "${seed}" \
        --adam-iterations "${ADAM_ITERATIONS}" \
        --lbfgs-iterations "${LBFGS_ITERATIONS}" \
        --display-every "${DISPLAY_EVERY}" \
        --output-dir "${seed_output_dir}"
}

cd "${ROOT_DIR}"

python -c 'import torch; assert torch.cuda.is_available(), "CUDA is unavailable"; print(f"PyTorch device: cuda:0 ({torch.cuda.get_device_name(0)})")'

read -r -a SEED_LIST <<< "${SEEDS}"

for seed in "${SEED_LIST[@]}"; do
    case "${CASE}" in
        all)
            run_case boundary-layer "${seed}"
            run_case steep "${seed}"
            ;;
        boundary-layer|steep)
            run_case "${CASE}" "${seed}"
            ;;
        *)
            echo "Usage: $0 [all|boundary-layer|steep]" >&2
            exit 2
            ;;
    esac
done

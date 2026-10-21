#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

CUDA_DEVICE="${CUDA_DEVICE:-0}"
export CUDA_VISIBLE_DEVICES="${CUDA_DEVICE}"
export DDE_BACKEND=pytorch
export PYTHONPATH="${ROOT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"

ADAM_ITERATIONS="${ADAM_ITERATIONS:-20000}"
LBFGS_ITERATIONS="${LBFGS_ITERATIONS:-1000}"
DISPLAY_EVERY="${DISPLAY_EVERY:-1000}"
SEEDS="${SEEDS:-2027 2028 2029 2030 2031 2032 2033}"
OUTPUT_DIR="${OUTPUT_DIR:-runs/multiseed_modified_mlp_corrected/advection_diffusion_1d}"
REFERENCE_PATH="${REFERENCE_PATH:-${ROOT_DIR}/reference_data/advection_diffusion_kappa_0.01_series_reference.npz}"

cd "${ROOT_DIR}"

python -c 'import torch; assert torch.cuda.is_available(), "CUDA is unavailable"; print(f"PyTorch device: cuda:0 ({torch.cuda.get_device_name(0)})")'

if [[ ! -f "${REFERENCE_PATH}" ]]; then
    echo "Generating 1D advection-diffusion analytical reference: ${REFERENCE_PATH}"
    python generate_advection_diffusion_reference.py --output "${REFERENCE_PATH}"
fi

read -r -a SEED_LIST <<< "${SEEDS}"

for seed in "${SEED_LIST[@]}"; do
    seed_output_dir="${OUTPUT_DIR}/seed_${seed}"
    echo "Running 1D advection-diffusion seed=${seed}, output=${seed_output_dir}"
    python -m pinn_baselines.advection_diffusion_1d \
        --seed "${seed}" \
        --adam-iterations "${ADAM_ITERATIONS}" \
        --lbfgs-iterations "${LBFGS_ITERATIONS}" \
        --display-every "${DISPLAY_EVERY}" \
        --reference-path "${REFERENCE_PATH}" \
        --output-dir "${seed_output_dir}"
done

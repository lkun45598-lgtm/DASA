#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PYTHON:-/home/lz/miniconda3/envs/pytorch/bin/python}"

CUDA_DEVICES="${CUDA_DEVICES:-0 1 2 3 4 5 6 7}"
export DDE_BACKEND=pytorch
export PYTHONPATH="${ROOT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"

ADAM_ITERATIONS="${ADAM_ITERATIONS:-300000}"
LBFGS_ITERATIONS="${LBFGS_ITERATIONS:-0}"
DISPLAY_EVERY="${DISPLAY_EVERY:-1000}"
CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-10000}"
RESUME_MODE="${RESUME_MODE:-auto}"
SEEDS="${SEEDS:-0 1 2 3 4 5 6 7 8 9}"
OUTPUT_DIR="${OUTPUT_DIR:-runs/standard_pinn_ac_10seeds_20260812}"
REFERENCE_PATH="${REFERENCE_PATH:-${ROOT_DIR}/reference_data/allen_cahn_diffusivity_0.0001_spectral_reference.npz}"

cd "${ROOT_DIR}"

if [[ ! -f "${REFERENCE_PATH}" ]]; then
    echo "Generating Allen-Cahn diffusivity=0.0001 reference: ${REFERENCE_PATH}"
    "${PYTHON}" generate_allen_cahn_reference.py --output "${REFERENCE_PATH}"
fi

read -r -a SEED_LIST <<< "${SEEDS}"
read -r -a GPU_LIST <<< "${CUDA_DEVICES}"

case "${RESUME_MODE}" in
    auto|required|never) ;;
    *)
        echo "RESUME_MODE must be one of: auto, required, never" >&2
        exit 2
        ;;
esac

if (( ${#GPU_LIST[@]} == 0 || ${#GPU_LIST[@]} > 8 )); then
    echo "CUDA_DEVICES must contain between 1 and 8 GPU IDs" >&2
    exit 2
fi

run_worker() {
    local worker_index="$1"
    local gpu_id="$2"
    local seed_index seed seed_output_dir checkpoint_dir

    CUDA_VISIBLE_DEVICES="${gpu_id}" "${PYTHON}" -c \
        'import torch; assert torch.cuda.is_available(), "CUDA is unavailable"; print(f"Worker device: {torch.cuda.get_device_name(0)}")'

    for ((seed_index=worker_index; seed_index<${#SEED_LIST[@]}; seed_index+=${#GPU_LIST[@]})); do
        seed="${SEED_LIST[seed_index]}"
        seed_output_dir="${OUTPUT_DIR}/seed_${seed}"
        checkpoint_dir="${seed_output_dir}/checkpoints"
        echo "Running 1D Allen-Cahn seed=${seed}, physical_gpu=${gpu_id}, resume=${RESUME_MODE}, checkpoints=${checkpoint_dir}, output=${seed_output_dir}"
        CUDA_VISIBLE_DEVICES="${gpu_id}" "${PYTHON}" -m plain_pinn_baselines.allen_cahn_1d \
            --seed "${seed}" \
            --adam-iterations "${ADAM_ITERATIONS}" \
            --lbfgs-iterations "${LBFGS_ITERATIONS}" \
            --display-every "${DISPLAY_EVERY}" \
            --checkpoint-dir "${checkpoint_dir}" \
            --checkpoint-every "${CHECKPOINT_EVERY}" \
            --resume-mode "${RESUME_MODE}" \
            --reference-path "${REFERENCE_PATH}" \
            --output-dir "${seed_output_dir}"
    done
}

PIDS=()
for ((worker_index=0; worker_index<${#GPU_LIST[@]}; worker_index++)); do
    if (( worker_index >= ${#SEED_LIST[@]} )); then
        break
    fi
    run_worker "${worker_index}" "${GPU_LIST[worker_index]}" &
    PIDS+=("$!")
done

status=0
for pid in "${PIDS[@]}"; do
    if ! wait "${pid}"; then
        status=1
    fi
done
exit "${status}"

#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

GPU_IDS="0,1,2,3,4,5,6,7"
CPU_COUNT="32"
SEED_SPEC="0-9"
PYTHON_BIN="python3"
OUTPUT_ROOT="$PROJECT_ROOT/results/poisson_1d_pure_gpinn"
SMOKE_ONLY="0"

CASES=(
  poisson_1d_boundary_layer
  poisson_1d_steep
)
CASE_CSV="poisson_1d_boundary_layer,poisson_1d_steep"

usage() {
  echo "Usage: $0 [--gpus 0,1,...,7|cpu] [--cpus N] [--seeds 0-9]"
  echo "          [--python PATH] [--output-root DIR] [--smoke-only]"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --gpus) GPU_IDS="$2"; shift 2 ;;
    --cpus) CPU_COUNT="$2"; shift 2 ;;
    --seeds) SEED_SPEC="$2"; shift 2 ;;
    --python) PYTHON_BIN="$2"; shift 2 ;;
    --output-root) OUTPUT_ROOT="$2"; shift 2 ;;
    --smoke-only) SMOKE_ONLY="1"; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

SEEDS=()
IFS=',' read -r -a SEED_TOKENS <<< "$SEED_SPEC"
for token in "${SEED_TOKENS[@]}"; do
  if [[ "$token" =~ ^[0-9]+$ ]]; then
    SEEDS+=("$token")
  elif [[ "$token" =~ ^([0-9]+)-([0-9]+)$ ]]; then
    first_seed="${BASH_REMATCH[1]}"
    last_seed="${BASH_REMATCH[2]}"
    if (( first_seed > last_seed )); then
      echo "Seed range must be ascending: $token" >&2
      exit 2
    fi
    for ((seed=first_seed; seed<=last_seed; seed++)); do
      SEEDS+=("$seed")
    done
  else
    echo "Invalid seed or seed range: $token" >&2
    exit 2
  fi
done
if [[ "${#SEEDS[@]}" -eq 0 ]]; then
  echo "--seeds must contain at least one seed" >&2
  exit 2
fi
for ((left=0; left<${#SEEDS[@]}; left++)); do
  for ((right=left+1; right<${#SEEDS[@]}; right++)); do
    if [[ "${SEEDS[$left]}" == "${SEEDS[$right]}" ]]; then
      echo "Duplicate seed: ${SEEDS[$left]}" >&2
      exit 2
    fi
  done
done
SEED_CSV="$(IFS=,; echo "${SEEDS[*]}")"

if ! [[ "$CPU_COUNT" =~ ^[1-9][0-9]*$ ]]; then
  echo "--cpus must be a positive integer" >&2
  exit 2
fi

if [[ "$GPU_IDS" == "cpu" ]]; then
  GPU_LIST=(cpu)
else
  IFS=',' read -r -a GPU_LIST <<< "$GPU_IDS"
  if [[ "${#GPU_LIST[@]}" -eq 0 ]]; then
    echo "--gpus must contain at least one GPU index" >&2
    exit 2
  fi
  for gpu_id in "${GPU_LIST[@]}"; do
    if ! [[ "$gpu_id" =~ ^[0-9]+$ ]]; then
      echo "Invalid GPU index: $gpu_id" >&2
      exit 2
    fi
  done
fi

cd "$PROJECT_ROOT"
if [[ -n "${CONDA_PREFIX:-}" && -d "$CONDA_PREFIX/lib" ]]; then
  export LD_LIBRARY_PATH="$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi

WORKER_COUNT="${#GPU_LIST[@]}"
THREADS_PER_WORKER=$((CPU_COUNT / WORKER_COUNT))
if [[ "$THREADS_PER_WORKER" -lt 1 ]]; then
  THREADS_PER_WORKER=1
fi

TASK_CASES=()
TASK_SEEDS=()
for seed in "${SEEDS[@]}"; do
  for case_name in "${CASES[@]}"; do
    TASK_CASES+=("$case_name")
    TASK_SEEDS+=("$seed")
  done
done
TASK_COUNT="${#TASK_CASES[@]}"

mkdir -p "$OUTPUT_ROOT"
LOG_PATH="$OUTPUT_ROOT/pipeline.log"
exec > >(tee -a "$LOG_PATH") 2>&1

echo "[setup] protocol=pure_gpinn_hard_dirichlet_no_rar"
echo "[setup] GPUs=$GPU_IDS, seeds=$SEED_CSV, tasks=$TASK_COUNT, workers=$WORKER_COUNT, total_cpus=$CPU_COUNT, cpus_per_worker=$THREADS_PER_WORKER"
echo "[setup] output=$OUTPUT_ROOT"

run_phase() {
  local mode="$1"
  local pids=()
  local worker_index

  for ((worker_index=0; worker_index<WORKER_COUNT; worker_index++)); do
    (
      local assigned_gpu="${GPU_LIST[$worker_index]}"
      local task_index
      export OMP_NUM_THREADS="$THREADS_PER_WORKER"
      export MKL_NUM_THREADS="$THREADS_PER_WORKER"
      export OPENBLAS_NUM_THREADS="$THREADS_PER_WORKER"
      export NUMEXPR_NUM_THREADS="$THREADS_PER_WORKER"
      export MPLCONFIGDIR="$OUTPUT_ROOT/.mplconfig_worker_${worker_index}"
      mkdir -p "$MPLCONFIGDIR"

      if [[ "$assigned_gpu" == "cpu" ]]; then
        export CUDA_VISIBLE_DEVICES=""
        local device="cpu"
      else
        export CUDA_VISIBLE_DEVICES="$assigned_gpu"
        local device="auto"
      fi

      for ((task_index=worker_index; task_index<TASK_COUNT; task_index+=WORKER_COUNT)); do
        local case_name="${TASK_CASES[$task_index]}"
        local seed="${TASK_SEEDS[$task_index]}"
        echo "[$mode][worker=$worker_index][gpu=$assigned_gpu] starting $case_name seed=$seed"
        "$PYTHON_BIN" -m gpinn_baselines.run \
          --case "$case_name" \
          --mode "$mode" \
          --seed "$seed" \
          --device "$device" \
          --output-root "$OUTPUT_ROOT" \
          --resume \
          --skip-summary
        echo "[$mode][worker=$worker_index][gpu=$assigned_gpu] completed $case_name seed=$seed"
      done
    ) &
    pids+=("$!")
  done

  local status=0
  local pid
  for pid in "${pids[@]}"; do
    if ! wait "$pid"; then
      status=1
    fi
  done
  if [[ "$status" -ne 0 ]]; then
    echo "[$mode] At least one worker failed" >&2
    return 1
  fi

  "$PYTHON_BIN" -m gpinn_baselines.summary \
    --mode "$mode" \
    --output-root "$OUTPUT_ROOT"
  "$PYTHON_BIN" -m gpinn_baselines.validate \
    --level results \
    --mode "$mode" \
    --cases "$CASE_CSV" \
    --seeds "$SEED_CSV" \
    --output-root "$OUTPUT_ROOT"
}

echo "[preflight] checking environment, equations, hard boundaries and network"
"$PYTHON_BIN" - <<'PY'
import numpy
import openpyxl
import scipy
import torch

print(
    f"NumPy={numpy.__version__}, SciPy={scipy.__version__}, "
    f"PyTorch={torch.__version__}, openpyxl={openpyxl.__version__}"
)
print(f"CUDA available={torch.cuda.is_available()}, visible devices={torch.cuda.device_count()}")
PY
"$PYTHON_BIN" -m pytest -q tests/test_baselines.py -k \
  "poisson_1d_original_gpinn_protocol_and_hard_constraints or network_contract"
"$PYTHON_BIN" -m gpinn_baselines.validate --level static

echo "[smoke] running $TASK_COUNT Poisson case/seed tasks"
run_phase smoke
echo "[smoke] all tasks and saved metrics passed validation"

if [[ "$SMOKE_ONLY" == "1" ]]; then
  echo "[done] --smoke-only requested; full experiments were not started"
  exit 0
fi

echo "[full] smoke passed; starting the full 10-seed experiments"
run_phase full

echo "[statistics] writing the two-row mean and sample-standard-deviation table"
"$PYTHON_BIN" -m gpinn_baselines.aggregate \
  --mode full \
  --cases "$CASE_CSV" \
  --seeds "$SEED_CSV" \
  --output-root "$OUTPUT_ROOT"

echo "[done] $OUTPUT_ROOT/full/results_mean_std.xlsx"

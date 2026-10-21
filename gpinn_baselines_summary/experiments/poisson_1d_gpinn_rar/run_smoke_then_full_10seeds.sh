#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
GPU_IDS="0,1,2,3,4,5,6,7"
CPU_COUNT="32"
SEEDS="0-9"
PYTHON_BIN="python"
PROTOCOL="burgers_rar_transfer"
OUTPUT_ROOT=""
SMOKE_ONLY="0"

usage() {
  echo "Usage: $0 [--gpus 0,1,...,7|cpu] [--cpus N] [--seeds 0-9]"
  echo "          [--protocol burgers_rar_transfer|paper_gpinn|upstream_poisson]"
  echo "          [--python PATH] [--output-root DIR] [--smoke-only]"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --gpus) GPU_IDS="$2"; shift 2 ;;
    --cpus) CPU_COUNT="$2"; shift 2 ;;
    --seeds) SEEDS="$2"; shift 2 ;;
    --protocol) PROTOCOL="$2"; shift 2 ;;
    --python) PYTHON_BIN="$2"; shift 2 ;;
    --output-root) OUTPUT_ROOT="$2"; shift 2 ;;
    --smoke-only) SMOKE_ONLY="1"; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

if [[ -z "$OUTPUT_ROOT" ]]; then
  OUTPUT_ROOT="$PROJECT_ROOT/results/poisson_1d_${PROTOCOL}_10seeds"
fi

cd "$PROJECT_ROOT"
if [[ -n "${CONDA_PREFIX:-}" && -d "$CONDA_PREFIX/lib" ]]; then
  export LD_LIBRARY_PATH="$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi
mkdir -p "$OUTPUT_ROOT"
LOG_PATH="$OUTPUT_ROOT/pipeline.log"

# Keep one pipeline per result directory. This also protects direct invocations
# of this script, not only launches through start_10seeds_nohup.sh.
if command -v flock >/dev/null 2>&1; then
  exec 9>"$OUTPUT_ROOT/.pipeline.lock"
  if ! flock -n 9; then
    echo "Another pipeline is already using $OUTPUT_ROOT" >&2
    exit 1
  fi
fi

exec > >(tee -a "$LOG_PATH") 2>&1

echo "[setup] protocol=$PROTOCOL, network=6x128, GPUs=$GPU_IDS, CPUs=$CPU_COUNT, seeds=$SEEDS"
echo "[setup] output=$OUTPUT_ROOT"
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
"$PYTHON_BIN" -m pytest -q tests/test_poisson_1d_gpinn_rar.py

echo "[smoke] starting two cases x ten seeds"
"$PYTHON_BIN" -u -m experiments.poisson_1d_gpinn_rar.run_10seeds \
  --mode smoke \
  --gpus "$GPU_IDS" \
  --cpus "$CPU_COUNT" \
  --seeds "$SEEDS" \
  --protocol "$PROTOCOL" \
  --output-root "$OUTPUT_ROOT"

if [[ "$SMOKE_ONLY" == "1" ]]; then
  echo "[done] smoke passed; full was not requested"
  exit 0
fi

echo "[full] every smoke result passed; starting full gPINN+RAR"
"$PYTHON_BIN" -u -m experiments.poisson_1d_gpinn_rar.run_10seeds \
  --mode full \
  --gpus "$GPU_IDS" \
  --cpus "$CPU_COUNT" \
  --seeds "$SEEDS" \
  --protocol "$PROTOCOL" \
  --output-root "$OUTPUT_ROOT"

echo "[done] $OUTPUT_ROOT/full/results_mean_std.xlsx"

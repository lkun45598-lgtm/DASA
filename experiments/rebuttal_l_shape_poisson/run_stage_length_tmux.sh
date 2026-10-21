#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT="$ROOT/run_stage_length_sensitivity.sh"
SESSION="${SESSION:-stage_length_sensitivity}"
RUN_ROOT="${RUN_ROOT:-$ROOT/results_stage_length_sensitivity}"
PYTHON="${PYTHON:-/home/lz/miniconda3/envs/pytorch/bin/python}"
SEEDS="${SEEDS:-0 1 2}"
if [[ "$#" -gt 0 ]]; then
  GPU_IDS=("$@")
else
  GPU_IDS=(0 1 2 3 4 5 6 7)
fi

if [[ "${1:-}" == "--status" ]]; then
  find "$RUN_ROOT" -name metrics.json | wc -l
  exit 0
fi
if ! command -v tmux >/dev/null 2>&1; then
  echo "tmux is not installed" >&2
  exit 1
fi
if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "tmux session already exists: $SESSION" >&2
  exit 1
fi

mkdir -p "$RUN_ROOT"
tmux new-session -d -s "$SESSION" -n monitor "watch -n 30 'find "$RUN_ROOT" -name metrics.json | wc -l'"

mapfile -t CONFIG_FILES < <(find "$ROOT/configs/stage_length" -maxdepth 1 -name '*.json' -print | sort)
gpu_count=${#GPU_IDS[@]}
for index in "${!CONFIG_FILES[@]}"; do
  gpu="${GPU_IDS[$((index % gpu_count))]}"
  config="${CONFIG_FILES[$index]}"
  command="CUDA_VISIBLE_DEVICES=$gpu PYTHON=$PYTHON SEEDS='$SEEDS' RESULTS='$RUN_ROOT' CONFIGS='$config' DEVICE=cuda:0 bash '$SCRIPT'"
  tmux new-window -t "$SESSION" -n "gpu${gpu}_cfg${index}" "$command"
done

echo "Started tmux session $SESSION on GPUs: ${GPU_IDS[*]}"
echo "Attach: tmux attach -t $SESSION"

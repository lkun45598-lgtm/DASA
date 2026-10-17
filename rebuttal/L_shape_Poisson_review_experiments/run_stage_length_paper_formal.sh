#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PYTHON:-/home/lz/miniconda3/envs/pytorch/bin/python}"
RUNNER="$ROOT/L_SHAPED_METHODS_FOR_GPT/refactored_dasa_pinn/src/lshape_experiment.py"
CONFIG_DIR="$ROOT/configs/stage_length_paper"
DATA="$ROOT/data/Lpoission_more.mat"
RESULTS="${RESULTS:-$ROOT/results_stage_length_paper_formal}"
SEEDS="${SEEDS:-0 1 2 3 4 5 6 7 8 9}"
QUEUE="$RESULTS/.queue"
PENDING="$QUEUE/pending.tsv"
RUNNING="$QUEUE/running.tsv"
COMPLETED="$QUEUE/completed.tsv"
FAILED="$QUEUE/failed.tsv"
LOCK="$QUEUE/queue.lock"

init_queue() {
  mkdir -p "$QUEUE"
  if [[ -f "$QUEUE/initialized" ]]; then
    echo "[queue] already initialized: $QUEUE"
    return
  fi
  : > "$PENDING"
  : > "$RUNNING"
  : > "$COMPLETED"
  : > "$FAILED"
  touch "$LOCK"
  for config in "$CONFIG_DIR"/paper_stage*.json; do
    experiment="$(basename "${config%.json}")"
    for seed in $SEEDS; do
      if [[ ! -f "$RESULTS/$experiment/seed_$seed/metrics.json" ]]; then
        printf '%s\t%s\t0\n' "$config" "$seed" >> "$PENDING"
      fi
    done
  done
  wc -l < "$PENDING" > "$QUEUE/total_runs"
  date -Is > "$QUEUE/initialized"
  echo "[queue] initialized with $(wc -l < "$PENDING") runs"
}

worker() {
  local gpu="$1" config seed attempt experiment run_dir temporary status
  local active_config="" active_seed="" active_attempt=""
  mkdir -p "$QUEUE"
  touch "$PENDING" "$RUNNING" "$COMPLETED" "$FAILED" "$LOCK"

  while true; do
    exec 9>"$LOCK"
    flock -x 9
    if [[ -s "$PENDING" ]]; then
      IFS=$'\t' read -r config seed attempt < "$PENDING"
      temporary="$PENDING.tmp.$$"
      sed '1d' "$PENDING" > "$temporary"
      mv "$temporary" "$PENDING"
      printf '%s\t%s\t%s\t%s\t%s\n' "$config" "$seed" "$attempt" "$gpu" "$(date -Is)" >> "$RUNNING"
      active_config="$config"
      active_seed="$seed"
      active_attempt="$attempt"
      flock -u 9
      exec 9>&-
    elif [[ -s "$RUNNING" ]]; then
      flock -u 9
      exec 9>&-
      sleep 10
      continue
    else
      flock -u 9
      exec 9>&-
      break
    fi

    experiment="$(basename "${config%.json}")"
    run_dir="$RESULTS/$experiment/seed_$seed"
    mkdir -p "$run_dir"
    if [[ -f "$run_dir/metrics.json" ]]; then
      status=0
    else
      set +e
      CUDA_VISIBLE_DEVICES="$gpu" CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONUNBUFFERED=1 \
        "$PYTHON" "$RUNNER" --config "$config" --seed "$seed" --device cuda:0 \
        --data "$DATA" --output-root "$RESULTS" >> "$run_dir/train.log" 2>&1
      status=$?
      set -e
    fi

    exec 9>"$LOCK"
    flock -x 9
    temporary="$RUNNING.tmp.$$"
    awk -F '\t' -v c="$config" -v s="$seed" -v a="$attempt" \
      '!($1 == c && $2 == s && $3 == a)' "$RUNNING" > "$temporary"
    mv "$temporary" "$RUNNING"
    if [[ "$status" -eq 0 && -f "$run_dir/metrics.json" ]]; then
      printf '%s\t%s\t%s\t%s\t%s\n' "$config" "$seed" "$attempt" "$gpu" "$(date -Is)" >> "$COMPLETED"
    elif [[ "$attempt" -lt 2 ]]; then
      printf '%s\t%s\t%s\n' "$config" "$seed" "$((attempt + 1))" >> "$PENDING"
    else
      printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$config" "$seed" "$attempt" "$gpu" "$(date -Is)" "$status" >> "$FAILED"
    fi
    flock -u 9
    exec 9>&-
    active_config=""
    active_seed=""
    active_attempt=""
  done
  echo "[$(date -Is)] worker GPU $gpu finished"
}

status_queue() {
  local total=0
  [[ -f "$QUEUE/total_runs" ]] && total="$(< "$QUEUE/total_runs")"
  printf 'metrics: %s/%s\n' "$(find "$RESULTS" -name metrics.json -not -path '*/.queue/*' | wc -l)" "$total"
  printf 'pending: %s\n' "$(wc -l < "$PENDING" 2>/dev/null || echo 0)"
  printf 'running: %s\n' "$(wc -l < "$RUNNING" 2>/dev/null || echo 0)"
  printf 'failed:  %s\n' "$(wc -l < "$FAILED" 2>/dev/null || echo 0)"
}

case "${1:-}" in
  init) init_queue ;;
  worker) worker "${2:?GPU id required}" ;;
  status) status_queue ;;
  aggregate)
    "$PYTHON" "$ROOT/src/aggregate_stage_length.py" --results "$RESULTS" --out "$RESULTS/summaries"
    ;;
  *) echo "usage: $0 {init|worker GPU|status|aggregate}" >&2; exit 2 ;;
esac

#!/usr/bin/env bash
set -euo pipefail

project_dir=/home/whn/hpy_pinns
result_root="$project_dir/results/table1_hpvpinn_poisson_boundary_prefined_20260811"
log="$project_dir/runs/table1_hpvpinn_poisson_boundary_prefined_20260811/summary_watch.log"
seeds=(1234 1235 1236 1237 1238 1239 1240 1241 1242 1243)

mkdir -p "$(dirname "$log")"
echo "[$(date -Is)] waiting for 10 p-refined boundary runs" > "$log"

while true; do
  completed=0
  for seed in "${seeds[@]}"; do
    metrics="$result_root/seed_${seed}/final/metrics.json"
    config="$result_root/seed_${seed}/final/config.json"
    if [[ -s "$metrics" && -s "$config" ]]; then
      completed=$((completed + 1))
    fi
  done
  echo "[$(date -Is)] completed=$completed/10" >> "$log"
  if [[ "$completed" == 10 ]]; then
    break
  fi
  sleep 30
done

/home/whn/miniconda3/envs/nan717/bin/python \
  "$project_dir/summarize_hpvpinn_multiseed.py" >> "$log" 2>&1
echo "[$(date -Is)] final summary updated" >> "$log"

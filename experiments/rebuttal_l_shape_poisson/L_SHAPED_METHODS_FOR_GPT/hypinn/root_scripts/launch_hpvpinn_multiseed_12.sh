#!/usr/bin/env bash
set -euo pipefail

project_dir=/home/whn/hpy_pinns
run_root="$project_dir/runs/table1_hpvpinn_multiseed_20260811"
queue_dir="$run_root/queue"
worker="$project_dir/run_hpvpinn_multiseed_worker.sh"
result_root="$project_dir/results/table1_hpvpinn_multiseed_20260811"

mkdir -p "$queue_dir" "$run_root/task_logs" "$result_root"

if [[ ! -f "$queue_dir/initialized" ]]; then
  : > "$queue_dir/pending.tsv"
  : > "$queue_dir/running.tsv"
  : > "$queue_dir/completed.tsv"
  : > "$queue_dir/failed.tsv"

  problems=(
    allen_cahn_1d
    poisson_2d_steep
    poisson_1d_steep
    advection_diffusion_1d
    burgers_1d
    poisson_1d_boundary
    poisson_lshape
    cavity_re100
    advection_discontinuous_1d
    helmholtz_2d
    poisson_2d_peak
    helmholtz_1d
  )
  seeds=(1235 1236 1237 1238 1239 1240 1241 1242 1243)

  for seed in "${seeds[@]}"; do
    for problem in "${problems[@]}"; do
      printf '%s\t%s\n' "$problem" "$seed" >> "$queue_dir/pending.tsv"
    done
  done

  {
    printf 'seed\trole\n'
    printf '1234\texisting_fixed_protocol_run\n'
    for seed in "${seeds[@]}"; do
      printf '%s\tnew_queue_run\n' "$seed"
    done
  } > "$queue_dir/seeds.tsv"

  printf '%s\n' "$(date -Is)" > "$queue_dir/initialized"
fi

for physical_gpu in 5 6 7; do
  session="hpv_ms12_gpu${physical_gpu}"
  if ! tmux has-session -t "$session" 2>/dev/null; then
    tmux new-session -d -s "$session" \
      "cd '$project_dir' && exec '$worker' '$physical_gpu' '$queue_dir' >> '$run_root/worker_gpu${physical_gpu}.log' 2>&1"
  fi
done

aux_session=hpv_ms12_gpu7b
if ! tmux has-session -t "$aux_session" 2>/dev/null; then
  tmux new-session -d -s "$aux_session" \
    "cd '$project_dir' && exec '$worker' '7' '$queue_dir' >> '$run_root/worker_gpu7b.log' 2>&1"
fi

tmux list-sessions -F '#{session_name}\t#{pane_pid}' | sort

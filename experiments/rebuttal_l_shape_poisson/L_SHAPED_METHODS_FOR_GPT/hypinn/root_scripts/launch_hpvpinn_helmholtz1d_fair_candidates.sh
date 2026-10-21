#!/usr/bin/env bash
set -euo pipefail

project_dir=/home/whn/hpy_pinns
python_bin=/home/whn/miniconda3/envs/nan717/bin/python
runner="$project_dir/hpvpinn_experiments.py"
result_root="$project_dir/results/helmholtz1d_fair_candidates_20260811"
run_root="$project_dir/runs/helmholtz1d_fair_candidates_20260811"
socket=hpv_h1_fair

mkdir -p "$result_root" "$run_root"

launch() {
  local name=$1
  local gpu=$2
  shift 2
  local output="$result_root/$name"
  local log="$run_root/$name.log"
  local session="h1_${name}"

  if [[ -s "$output/metrics.json" ]]; then
    echo "completed $name"
    return
  fi
  if tmux -L "$socket" has-session -t "$session" 2>/dev/null; then
    echo "running $name"
    return
  fi

  local command
  command="cd $project_dir && exec env CUDA_VISIBLE_DEVICES=$gpu CUBLAS_WORKSPACE_CONFIG=:4096:8 \
    $python_bin -u $runner --problem helmholtz_1d --seed 1234 --device cuda \
    --dtype float64 --iterations 21001 --checkpoint-every 2000 --log-every 100 \
    --output-dir $output --variational-form strong --boundary-weight 100 \
    --learning-rate 0.001 --lr-step 2000 --lr-gamma 0.9 $* > $log 2>&1"
  tmux -L "$socket" new-session -d -s "$session" "$command"
  echo "started $name gpu=$gpu log=$log"
}

launch full_h10_p22q32 1 \
  --mesh-variant baseline --test-basis legendre_full --n-test 22 --n-quad 32
launch bubble_h20_p20q32 3 \
  --mesh-variant refined --test-basis legendre_bubble --n-test 20 --n-quad 32
launch full_h20_p12q24 4 \
  --mesh-variant refined --test-basis legendre_full --n-test 12 --n-quad 24


#!/usr/bin/env bash
set -euo pipefail

project_dir=/home/whn/hpy_pinns
python_bin=/home/whn/miniconda3/envs/nan717/bin/python
runner="$project_dir/hpvpinn_experiments.py"
result_root="$project_dir/results/discontinuous_advection_fair_candidates_20260811"
run_root="$project_dir/runs/discontinuous_advection_fair_candidates_20260811"
socket=hpv_ad_fair

mkdir -p "$result_root" "$run_root"

launch() {
  local name=$1
  local gpu=$2
  shift 2
  local output="$result_root/$name"
  local log="$run_root/$name.log"
  local session="ad_${name}"

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
    $python_bin -u $runner --problem advection_discontinuous_1d --seed 1234 \
    --device cuda --dtype float64 --iterations 26001 --checkpoint-every 2000 \
    --log-every 100 --output-dir $output --variational-form strong \
    --mesh-variant baseline --n-test 5 --n-quad 10 --boundary-weight 100 \
    --initial-weight 100 --learning-rate 0.001 --lr-step 2000 --lr-gamma 0.9 \
    $* > $log 2>&1"
  tmux -L "$socket" new-session -d -s "$session" "$command"
  echo "started $name gpu=$gpu log=$log"
}

launch standard_full_p5q10 1 \
  --coordinate-transform standard --test-basis legendre_full
launch characteristic_bubble_p5q10 3 \
  --coordinate-transform characteristic --test-basis legendre_bubble
launch characteristic_full_p5q10 4 \
  --coordinate-transform characteristic --test-basis legendre_full
launch standard_full_p5q10_avg 5 \
  --coordinate-transform standard --test-basis legendre_full \
  --projection-normalization element_average
launch characteristic_bubble_p5q10_avg 6 \
  --coordinate-transform characteristic --test-basis legendre_bubble \
  --projection-normalization element_average
launch characteristic_full_p5q10_avg 7 \
  --coordinate-transform characteristic --test-basis legendre_full \
  --projection-normalization element_average
launch characteristic_bubble_p5q10_avg_refined_ic 1 \
  --coordinate-transform characteristic --test-basis legendre_bubble \
  --projection-normalization element_average --initial-sampling interface_refined
launch characteristic_full_p5q10_avg_refined_ic 3 \
  --coordinate-transform characteristic --test-basis legendre_full \
  --projection-normalization element_average --initial-sampling interface_refined

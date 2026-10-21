#!/usr/bin/env bash
set -euo pipefail

project_dir=/home/whn/hpy_pinns
python_bin=/home/whn/miniconda3/envs/nan717/bin/python
runner="$project_dir/hpvpinn_experiments.py"
result_root="$project_dir/results/discontinuous_advection_characteristic_multiseed_20260811"
run_root="$project_dir/runs/discontinuous_advection_characteristic_multiseed_20260811"
socket=hpv_ad_multi
seeds=(1234 1235 1236 1237 1238 1239 1240 1241 1242 1243)
gpus=(0 1 2 3 4 5 6 7 1 3)

mkdir -p "$result_root" "$run_root"

for index in "${!seeds[@]}"; do
  seed=${seeds[$index]}
  gpu=${gpus[$index]}
  output="$result_root/seed_${seed}"
  log="$run_root/seed_${seed}.log"
  session="ad_multi_${seed}"

  if [[ -s "$output/metrics.json" ]]; then
    echo "completed seed=$seed"
    continue
  fi
  if tmux -L "$socket" has-session -t "$session" 2>/dev/null; then
    echo "running seed=$seed gpu=$gpu"
    continue
  fi

  command="cd $project_dir && exec env CUDA_VISIBLE_DEVICES=$gpu CUBLAS_WORKSPACE_CONFIG=:4096:8 \
    $python_bin -u $runner --problem advection_discontinuous_1d --seed $seed \
    --device cuda --dtype float64 --iterations 26001 --checkpoint-every 2000 \
    --log-every 100 --output-dir $output --variational-form strong \
    --mesh-variant baseline --coordinate-transform characteristic \
    --test-basis legendre_bubble --n-test 5 --n-quad 10 \
    --boundary-weight 100 --initial-weight 100 \
    --projection-normalization element_average --initial-sampling uniform \
    --learning-rate 0.001 --lr-step 2000 --lr-gamma 0.9 > $log 2>&1"
  tmux -L "$socket" new-session -d -s "$session" "$command"
  echo "started seed=$seed gpu=$gpu log=$log"
done


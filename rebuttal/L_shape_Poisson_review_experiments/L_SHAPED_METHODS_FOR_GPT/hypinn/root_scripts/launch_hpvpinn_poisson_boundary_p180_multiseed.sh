#!/usr/bin/env bash
set -euo pipefail

project_dir=/home/whn/hpy_pinns
python_bin=/home/whn/miniconda3/envs/nan717/bin/python
runner="$project_dir/hpvpinn_experiments.py"
result_root="$project_dir/results/table1_hpvpinn_poisson_boundary_prefined_20260811"
run_root="$project_dir/runs/table1_hpvpinn_poisson_boundary_prefined_20260811"
seeds=(1234 1235 1236 1237 1238 1239 1240 1241 1242 1243)
gpus=(0 1 2 3 4 5 6 7 0 1)

mkdir -p "$result_root" "$run_root"

for index in "${!seeds[@]}"; do
  seed=${seeds[$index]}
  gpu=${gpus[$index]}
  source="$result_root/seed_${seed}/final/checkpoint_latest.pt"
  source_metrics="$(dirname "$source")/metrics.json"
  output="$result_root/seed_${seed}/p180_q200"
  log="$run_root/seed_${seed}_p180_q200.log"
  session="hpv_pb_p180_${seed}"

  if [[ ! -s "$source" || ! -s "$source_metrics" ]]; then
    echo "missing completed p100/q140 source for seed $seed: $source" >&2
    exit 1
  fi
  if [[ -s "$output/metrics.json" ]]; then
    echo "completed seed=$seed output=$output"
    continue
  fi
  if tmux -L hpvpinn_pb180 has-session -t "$session" 2>/dev/null; then
    echo "running seed=$seed gpu=$gpu session=$session"
    continue
  fi

  command="cd $project_dir && exec env CUDA_VISIBLE_DEVICES=$gpu CUBLAS_WORKSPACE_CONFIG=:4096:8 $python_bin -u $runner \
    --problem poisson_1d_boundary --seed $seed --device cuda --dtype float64 \
    --iterations 30001 --checkpoint-every 2000 --log-every 100 \
    --output-dir $output --variational-form strong --mesh-variant baseline \
    --n-test 180 --n-quad 200 --boundary-weight 1 \
    --learning-rate 1e-5 --lr-step 2000 --lr-gamma 1.0 \
    --init-checkpoint $source > $log 2>&1"
  tmux -L hpvpinn_pb180 new-session -d -s "$session" "$command"
  echo "started seed=$seed gpu=$gpu session=$session log=$log"
done

#!/usr/bin/env bash
set -euo pipefail

project_dir=/home/whn/hpy_pinns
python_bin=/home/whn/miniconda3/envs/nan717/bin/python
runner="$project_dir/hpvpinn_experiments.py"
result_root="$project_dir/results/table1_hpvpinn_poisson_boundary_prefined_20260811"
run_root="$project_dir/runs/table1_hpvpinn_poisson_boundary_prefined_20260811"

mkdir -p "$result_root" "$run_root"

source_checkpoint() {
  local seed=$1
  if [[ "$seed" == 1234 ]]; then
    printf '%s\n' "$project_dir/results/table1_hpvpinn_optimized/poisson_1d_boundary/adam_continuation_lr1e-5_seed_1234/checkpoint_latest.pt"
  else
    printf '%s\n' "$project_dir/results/table1_hpvpinn_multiseed_20260811/poisson_1d_boundary/seed_${seed}/final/checkpoint_latest.pt"
  fi
}

seeds=(1234 1235 1236 1237 1238 1239 1240 1241 1242 1243)
gpus=(0 1 2 3 4 5 6 7 0 1)

for index in "${!seeds[@]}"; do
  seed=${seeds[$index]}
  gpu=${gpus[$index]}
  source=$(source_checkpoint "$seed")
  source_metrics="$(dirname "$source")/metrics.json"
  output="$result_root/seed_${seed}/final"
  log="$run_root/seed_${seed}.log"
  session="hpv_pb_p100_${seed}"

  if [[ ! -s "$source" || ! -s "$source_metrics" ]]; then
    echo "missing completed source for seed $seed: $source" >&2
    exit 1
  fi
  if [[ -s "$output/metrics.json" ]]; then
    echo "completed seed=$seed output=$output"
    continue
  fi
  if tmux -L hpvpinn_pb has-session -t "$session" 2>/dev/null; then
    echo "running seed=$seed gpu=$gpu session=$session"
    continue
  fi

  command="cd $project_dir && exec env CUDA_VISIBLE_DEVICES=$gpu CUBLAS_WORKSPACE_CONFIG=:4096:8 $python_bin -u $runner \
    --problem poisson_1d_boundary --seed $seed --device cuda --dtype float64 \
    --iterations 30001 --checkpoint-every 2000 --log-every 100 \
    --output-dir $output --variational-form strong --mesh-variant baseline \
    --n-test 100 --n-quad 140 --boundary-weight 1 \
    --learning-rate 1e-5 --lr-step 2000 --lr-gamma 1.0 \
    --init-checkpoint $source > $log 2>&1"
  tmux -L hpvpinn_pb new-session -d -s "$session" "$command"
  echo "started seed=$seed gpu=$gpu session=$session log=$log"
done

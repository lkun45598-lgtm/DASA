#!/usr/bin/env bash
set -euo pipefail

project_dir=/home/whn/hpy_pinns
python_bin=/home/whn/miniconda3/envs/nan717/bin/python
runner="$project_dir/hpvpinn_experiments.py"
result_root="$project_dir/results/table1_hpvpinn_diffusion_reaction_candidates_20260811"
run_root="$project_dir/runs/table1_hpvpinn_diffusion_reaction_candidates_20260811"
seed=1234

mkdir -p "$result_root" "$run_root"

launch_candidate() {
  local label=$1
  local physical_gpu=$2
  shift 2
  local output_dir="$result_root/$label"
  local log_file="$run_root/$label.log"
  local session="hpv_dr_${label}"
  if [[ -s "$output_dir/metrics.json" ]] || tmux has-session -t "$session" 2>/dev/null; then
    return
  fi
  tmux new-session -d -s "$session" \
    "cd '$project_dir' && exec env CUDA_VISIBLE_DEVICES='$physical_gpu' CUBLAS_WORKSPACE_CONFIG=:4096:8 \
      '$python_bin' -u '$runner' --problem diffusion_reaction_1d --seed '$seed' \
      --device cuda --dtype float64 --iterations 21001 --checkpoint-every 2000 --log-every 100 \
      --output-dir '$output_dir' $* > '$log_file' 2>&1"
}

launch_candidate strong_temporal_full_p5q10 5 \
  --variational-form strong --mesh-variant baseline --temporal-test-basis full \
  --n-test 5 --n-quad 10 --boundary-weight 100 --initial-weight 100 \
  --learning-rate 0.001 --lr-step 2000 --lr-gamma 0.9

launch_candidate p1_temporal_full_p5q10 6 \
  --variational-form p1 --mesh-variant baseline --temporal-test-basis full \
  --n-test 5 --n-quad 10 --boundary-weight 100 --initial-weight 100 \
  --learning-rate 0.001 --lr-step 2000 --lr-gamma 0.9

launch_candidate r1_single_domain_p20q80 7 \
  --variational-form r1 --mesh-variant paper \
  --test-basis chebyshev_full --quadrature-rule clenshaw_curtis \
  --n-test 20 --n-quad 80 --boundary-weight 100 --initial-weight 100 \
  --learning-rate 0.001 --lr-step 2000 --lr-gamma 0.9

launch_candidate p1_temporal_full_normalized_p5q10 6 \
  --variational-form p1 --mesh-variant baseline --temporal-test-basis full \
  --projection-normalization element_average \
  --n-test 5 --n-quad 10 --boundary-weight 100 --initial-weight 100 \
  --learning-rate 0.0001 --lr-step 2000 --lr-gamma 0.9

p1_source="$result_root/p1_temporal_full_p5q10/checkpoint_latest.pt"
if [[ -s "$p1_source" && -s "$result_root/p1_temporal_full_p5q10/metrics.json" ]]; then
  launch_candidate p1_normalized_continuation_p5q10 5 \
    --variational-form p1 --mesh-variant baseline --temporal-test-basis full \
    --projection-normalization element_average \
    --n-test 5 --n-quad 10 --boundary-weight 100 --initial-weight 100 \
    --learning-rate 0.0001 --lr-step 2000 --lr-gamma 0.9 \
    --init-checkpoint "$p1_source"

  launch_candidate p1_spacetime_ibp_continuation_p5q10 7 \
    --variational-form p1 --mesh-variant baseline --temporal-test-basis full \
    --integrate-time-by-parts \
    --n-test 5 --n-quad 10 --boundary-weight 100 --initial-weight 100 \
    --learning-rate 0.0001 --lr-step 2000 --lr-gamma 0.9 \
    --init-checkpoint "$p1_source"

  launch_candidate p1_prefined_p10q20_continuation 6 \
    --variational-form p1 --mesh-variant baseline --temporal-test-basis full \
    --n-test 10 --n-quad 20 --boundary-weight 100 --initial-weight 100 \
    --learning-rate 0.0001 --lr-step 2000 --lr-gamma 0.9 \
    --init-checkpoint "$p1_source"
fi

tmux list-sessions -F '#{session_name}\t#{pane_pid}' | rg '^hpv_dr_' | sort

#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 3 ]]; then
  echo "usage: $0 PROBLEM SEED PHYSICAL_GPU" >&2
  exit 2
fi

problem=$1
seed=$2
physical_gpu=$3

project_dir=/home/whn/hpy_pinns
python_bin=/home/whn/miniconda3/envs/nan717/bin/python
runner="$project_dir/hpvpinn_experiments.py"
result_root="$project_dir/results/table1_hpvpinn_multiseed_20260811"
seed_dir="$result_root/$problem/seed_$seed"

export CUDA_VISIBLE_DEVICES="$physical_gpu"
export CUBLAS_WORKSPACE_CONFIG=:4096:8

mkdir -p "$seed_dir"

run_training() {
  local output_dir=$1
  shift
  if [[ -s "$output_dir/metrics.json" ]]; then
    echo "[$(date -Is)] skip completed stage: $output_dir"
    return
  fi
  mkdir -p "$output_dir"
  echo "[$(date -Is)] start stage: $output_dir"
  "$python_bin" -u "$runner" \
    --problem "$problem" \
    --seed "$seed" \
    --device cuda \
    --dtype float64 \
    --checkpoint-every 2000 \
    --log-every 100 \
    --output-dir "$output_dir" \
    "$@"
  echo "[$(date -Is)] finish stage: $output_dir"
}

run_burgers_refinement() {
  local source_dir=$1
  local mesh_file=$2
  local report_file=$3
  if [[ -s "$mesh_file" && -s "$report_file" ]]; then
    echo "[$(date -Is)] skip completed Burgers refinement: $mesh_file"
    return
  fi
  "$python_bin" -u "$project_dir/refine_burgers_spatial_strip.py" \
    "$source_dir" \
    --device cuda \
    --theta 0.5 \
    --x-order 201 \
    --t-order 101 \
    --output-mesh "$mesh_file" \
    --output-report "$report_file"
}

run_allen_refinement() {
  local source_dir=$1
  local time_elements=$2
  local mesh_file=$3
  local report_file=$4
  if [[ -s "$mesh_file" && -s "$report_file" ]]; then
    echo "[$(date -Is)] skip completed Allen-Cahn refinement: $mesh_file"
    return
  fi
  "$python_bin" -u "$project_dir/refine_allen_cahn_spatial_bands.py" \
    "$source_dir" \
    --device cuda \
    --theta 0.5 \
    --x-order 201 \
    --t-order 101 \
    --time-elements "$time_elements" \
    --output-mesh "$mesh_file" \
    --output-report "$report_file"
}

echo "[$(date -Is)] task start problem=$problem seed=$seed physical_gpu=$physical_gpu"

case "$problem" in
  poisson_1d_boundary)
    stage1="$seed_dir/stage1_p60q80"
    run_training "$stage1" \
      --iterations 40001 --variational-form strong --mesh-variant baseline \
      --n-test 60 --n-quad 80 --boundary-weight 1 \
      --learning-rate 0.001 --lr-step 2000 --lr-gamma 0.9
    run_training "$seed_dir/final" \
      --iterations 20001 --variational-form strong --mesh-variant baseline \
      --n-test 60 --n-quad 80 --boundary-weight 1 \
      --learning-rate 1e-5 --lr-step 2000 --lr-gamma 1.0 \
      --init-checkpoint "$stage1/checkpoint_latest.pt"
    ;;

  poisson_1d_steep)
    stage1="$seed_dir/stage1_p60q80"
    stage2="$seed_dir/stage2_p60q80_lr1e5"
    stage3="$seed_dir/stage3_p100q140"
    run_training "$stage1" \
      --iterations 40001 --variational-form strong --mesh-variant baseline \
      --n-test 60 --n-quad 80 --boundary-weight 1 \
      --learning-rate 0.001 --lr-step 2000 --lr-gamma 0.9
    run_training "$stage2" \
      --iterations 20001 --variational-form strong --mesh-variant baseline \
      --n-test 60 --n-quad 80 --boundary-weight 1 \
      --learning-rate 1e-5 --lr-step 2000 --lr-gamma 1.0 \
      --init-checkpoint "$stage1/checkpoint_latest.pt"
    run_training "$stage3" \
      --iterations 30001 --variational-form strong --mesh-variant baseline \
      --n-test 100 --n-quad 140 --boundary-weight 1 \
      --learning-rate 1e-5 --lr-step 2000 --lr-gamma 1.0 \
      --init-checkpoint "$stage2/checkpoint_latest.pt"
    run_training "$seed_dir/final" \
      --iterations 30001 --variational-form strong --mesh-variant baseline \
      --n-test 180 --n-quad 200 --boundary-weight 1 \
      --learning-rate 1e-5 --lr-step 2000 --lr-gamma 1.0 \
      --init-checkpoint "$stage3/checkpoint_latest.pt"
    ;;

  helmholtz_1d)
    run_training "$seed_dir/final" \
      --iterations 21001 --variational-form strong --mesh-variant baseline \
      --n-test 20 --n-quad 32 --boundary-weight 100 \
      --learning-rate 0.001 --lr-step 2000 --lr-gamma 0.9
    ;;

  poisson_2d_peak)
    run_training "$seed_dir/final" \
      --iterations 21001 --variational-form strong --mesh-variant baseline \
      --n-test 5 --n-quad 10 --boundary-weight 100 \
      --learning-rate 0.001 --lr-step 2000 --lr-gamma 0.9
    ;;

  poisson_2d_steep)
    stage1="$seed_dir/stage1_p1_p5q10"
    stage2="$seed_dir/stage2_p1_p5q10_to100k"
    run_training "$stage1" \
      --iterations 21001 --variational-form p1 --mesh-variant official \
      --n-test 5 --n-quad 10 --boundary-weight 10 \
      --learning-rate 0.001 --lr-step 2000 --lr-gamma 1.0
    run_training "$stage2" \
      --iterations 79000 --variational-form p1 --mesh-variant official \
      --n-test 5 --n-quad 10 --boundary-weight 10 \
      --learning-rate 0.001 --lr-step 2000 --lr-gamma 1.0 \
      --init-checkpoint "$stage1/checkpoint_latest.pt"
    run_training "$seed_dir/final" \
      --iterations 30001 --variational-form p1 --mesh-variant official \
      --n-test 8 --n-quad 16 --boundary-weight 10 \
      --learning-rate 0.0001 --lr-step 2000 --lr-gamma 1.0 \
      --init-checkpoint "$stage2/checkpoint_latest.pt"
    ;;

  helmholtz_2d)
    run_training "$seed_dir/final" \
      --iterations 21001 --variational-form p1 --mesh-variant refined \
      --n-test 5 --n-quad 10 --boundary-weight 100 \
      --learning-rate 0.001 --lr-step 2000 --lr-gamma 0.9
    ;;

  cavity_re100)
    run_training "$seed_dir/final" \
      --iterations 25001 --variational-form strong --mesh-variant baseline \
      --n-test 5 --n-quad 10 --boundary-weight 100 \
      --learning-rate 0.001 --lr-step 2000 --lr-gamma 0.9
    ;;

  burgers_1d)
    stage1="$seed_dir/stage1_single_domain_p20q80"
    mesh_file="$seed_dir/residual_spatial_strip.npy"
    report_file="$seed_dir/residual_spatial_strip_report.json"
    run_training "$stage1" \
      --iterations 36000 --variational-form r1 --mesh-variant paper \
      --test-basis chebyshev_full --quadrature-rule clenshaw_curtis \
      --n-test 20 --n-quad 80 --boundary-weight 1 --initial-weight 1 \
      --learning-rate 0.001 --lr-step 2000 --lr-gamma 0.9
    run_burgers_refinement "$stage1" "$mesh_file" "$report_file"
    run_training "$seed_dir/final" \
      --iterations 24000 --variational-form r1 --mesh-variant baseline \
      --test-basis chebyshev_full --quadrature-rule clenshaw_curtis \
      --n-test 20 --n-quad 80 --boundary-weight 1 --initial-weight 1 \
      --learning-rate 0.001 --lr-step 2000 --lr-gamma 0.9 \
      --elements-file "$mesh_file" \
      --init-checkpoint "$stage1/checkpoint_latest.pt"
    ;;

  allen_cahn_1d)
    stage1="$seed_dir/stage1_single_domain_p20q80"
    mesh1="$seed_dir/residual_spatial_bands.npy"
    report1="$seed_dir/residual_spatial_bands_report.json"
    stage2="$seed_dir/stage2_spatial_bands_p20q80_8001"
    mesh2="$seed_dir/residual_spacetime_h5x8.npy"
    report2="$seed_dir/residual_spacetime_h5x8_report.json"
    run_training "$stage1" \
      --iterations 180000 --variational-form r1 --mesh-variant paper \
      --test-basis chebyshev_full --quadrature-rule clenshaw_curtis \
      --n-test 20 --n-quad 80 --fourier-modes 1 \
      --boundary-weight 1 --initial-weight 100 \
      --learning-rate 0.001 --lr-step 2000 --lr-gamma 1.0
    run_allen_refinement "$stage1" 1 "$mesh1" "$report1"
    run_training "$stage2" \
      --iterations 8001 --variational-form r1 --mesh-variant baseline \
      --test-basis chebyshev_full --quadrature-rule clenshaw_curtis \
      --n-test 20 --n-quad 80 --fourier-modes 1 \
      --boundary-weight 1 --initial-weight 100 \
      --learning-rate 0.001 --lr-step 2000 --lr-gamma 1.0 \
      --elements-file "$mesh1" \
      --init-checkpoint "$stage1/checkpoint_latest.pt"
    run_allen_refinement "$stage2" 8 "$mesh2" "$report2"
    run_training "$seed_dir/final" \
      --iterations 20000 --variational-form r1 --mesh-variant baseline \
      --test-basis chebyshev_full --quadrature-rule clenshaw_curtis \
      --n-test 10 --n-quad 20 --fourier-modes 1 \
      --boundary-weight 1 --initial-weight 100 \
      --projection-normalization element_average \
      --learning-rate 0.0001 --lr-step 2000 --lr-gamma 0.9 \
      --elements-file "$mesh2" \
      --init-checkpoint "$stage2/checkpoint_latest.pt"
    ;;

  advection_diffusion_1d)
    stage1="$seed_dir/stage1_h4_p5q10_lr1e4"
    run_training "$stage1" \
      --iterations 21001 --variational-form p1 --mesh-variant official \
      --n-test 5 --n-quad 10 --boundary-weight 10 --initial-weight 10 \
      --constraint-aggregation combined --ade-kappa 0.01 \
      --learning-rate 0.0001 --lr-step 2000 --lr-gamma 1.0
    run_training "$seed_dir/final" \
      --iterations 79000 --variational-form p1 --mesh-variant official \
      --n-test 5 --n-quad 10 --boundary-weight 10 --initial-weight 10 \
      --constraint-aggregation combined --ade-kappa 0.01 \
      --learning-rate 1e-5 --lr-step 2000 --lr-gamma 1.0 \
      --init-checkpoint "$stage1/checkpoint_latest.pt"
    ;;

  poisson_lshape)
    run_training "$seed_dir/final" \
      --iterations 28001 --variational-form p1 --mesh-variant official \
      --n-test 8 --n-quad 16 --boundary-weight 10 \
      --learning-rate 0.001 --lr-step 2000 --lr-gamma 0.9
    ;;

  advection_discontinuous_1d)
    run_training "$seed_dir/final" \
      --iterations 26001 --variational-form strong --mesh-variant baseline \
      --n-test 5 --n-quad 10 --boundary-weight 100 \
      --learning-rate 0.001 --lr-step 2000 --lr-gamma 0.9
    ;;

  diffusion_reaction_1d)
    stage1="$seed_dir/stage1_p1_temporal_full_p5q10"
    run_training "$stage1" \
      --iterations 21001 --variational-form p1 --mesh-variant baseline \
      --temporal-test-basis full --n-test 5 --n-quad 10 \
      --boundary-weight 100 --initial-weight 100 \
      --learning-rate 0.001 --lr-step 2000 --lr-gamma 0.9
    run_training "$seed_dir/final" \
      --iterations 21001 --variational-form p1 --mesh-variant baseline \
      --temporal-test-basis full --n-test 10 --n-quad 20 \
      --boundary-weight 100 --initial-weight 100 \
      --learning-rate 0.0001 --lr-step 2000 --lr-gamma 0.9 \
      --init-checkpoint "$stage1/checkpoint_latest.pt"
    ;;

  *)
    echo "unsupported problem: $problem" >&2
    exit 2
    ;;
esac

echo "[$(date -Is)] task complete problem=$problem seed=$seed physical_gpu=$physical_gpu"

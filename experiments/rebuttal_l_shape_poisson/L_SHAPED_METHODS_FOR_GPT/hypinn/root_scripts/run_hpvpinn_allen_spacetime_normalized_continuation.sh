#!/usr/bin/env bash
set -euo pipefail

gpu=${1:-5}
project_dir=/home/whn/hpy_pinns
python_bin=/home/whn/miniconda3/envs/nan717/bin/python
source_dir="$project_dir/results/table1_hpvpinn_allen_paper_extension_r1_20260810/spatial_bands_p20q80"
result_root="$project_dir/results/table1_hpvpinn_allen_spacetime_normalized_20260810"
run_root="$project_dir/runs/table1_hpvpinn_allen_spacetime_normalized_20260810"
mesh_file="$result_root/residual_spacetime_h5x8.npy"
report_file="$result_root/residual_spacetime_h5x8_report.json"
output_dir="$result_root/p10q20_h5x8_lr1e4_seed_1234"

mkdir -p "$result_root" "$run_root" "$output_dir"

CUDA_VISIBLE_DEVICES="$gpu" "$python_bin" -u "$project_dir/refine_allen_cahn_spatial_bands.py" \
    "$source_dir" \
    --device cuda \
    --theta 0.5 \
    --x-order 201 \
    --t-order 101 \
    --time-elements 8 \
    --output-mesh "$mesh_file" \
    --output-report "$report_file" \
    >"$run_root/refine.log" 2>&1

CUDA_VISIBLE_DEVICES="$gpu" "$python_bin" -u "$project_dir/hpvpinn_experiments.py" \
    --problem allen_cahn_1d \
    --seed 1234 \
    --device cuda \
    --dtype float64 \
    --variational-form r1 \
    --test-basis chebyshev_full \
    --quadrature-rule clenshaw_curtis \
    --elements-file "$mesh_file" \
    --n-test 10 \
    --n-quad 20 \
    --fourier-modes 1 \
    --boundary-weight 1 \
    --initial-weight 100 \
    --projection-normalization element_average \
    --learning-rate 0.0001 \
    --lr-step 2000 \
    --lr-gamma 0.9 \
    --iterations 20000 \
    --checkpoint-every 2000 \
    --log-every 100 \
    --init-checkpoint "$source_dir/checkpoint_latest.pt" \
    --output-dir "$output_dir" \
    >"$run_root/train.log" 2>&1

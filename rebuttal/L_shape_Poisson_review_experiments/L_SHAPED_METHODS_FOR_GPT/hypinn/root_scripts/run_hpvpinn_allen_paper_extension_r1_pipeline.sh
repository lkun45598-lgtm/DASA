#!/usr/bin/env bash
set -euo pipefail

gpu=${1:-6}
project_dir=/home/whn/hpy_pinns
python_bin=/home/whn/miniconda3/envs/nan717/bin/python
result_root="$project_dir/results/table1_hpvpinn_allen_paper_extension_r1_20260810"
run_root="$project_dir/runs/table1_hpvpinn_allen_paper_extension_r1_20260810"
base_dir="$result_root/base_single_domain_p20q80"
refined_dir="$result_root/spatial_bands_p20q80"
mesh_file="$result_root/residual_spatial_bands.npy"
report_file="$result_root/spatial_band_report.json"
mkdir -p "$result_root" "$run_root"

common_args=(
    --problem allen_cahn_1d
    --seed 1234
    --device cuda
    --dtype float64
    --variational-form r1
    --test-basis chebyshev_full
    --quadrature-rule clenshaw_curtis
    --n-test 20
    --n-quad 80
    --fourier-modes 1
    --boundary-weight 1
    --initial-weight 100
    --learning-rate 0.001
    --lr-gamma 1.0
    --checkpoint-every 2000
    --log-every 100
)

echo "[$(date -Is)] Allen-Cahn paper-method extension R1 base stage on physical GPU $gpu"
CUDA_VISIBLE_DEVICES="$gpu" "$python_bin" -u "$project_dir/hpvpinn_experiments.py" \
    "${common_args[@]}" --mesh-variant paper --iterations 180000 --output-dir "$base_dir" \
    2>&1 | tee "$run_root/base.log"

echo "[$(date -Is)] Allen-Cahn residual-driven spatial bands"
CUDA_VISIBLE_DEVICES="$gpu" "$python_bin" -u "$project_dir/refine_allen_cahn_spatial_bands.py" \
    "$base_dir" --device cuda --theta 0.5 --x-order 201 --t-order 101 \
    --output-mesh "$mesh_file" --output-report "$report_file" \
    2>&1 | tee "$run_root/refine.log"

echo "[$(date -Is)] Allen-Cahn paper-method extension spatial-refined stage"
CUDA_VISIBLE_DEVICES="$gpu" "$python_bin" -u "$project_dir/hpvpinn_experiments.py" \
    "${common_args[@]}" --elements-file "$mesh_file" --iterations 120000 \
    --init-checkpoint "$base_dir/checkpoint_latest.pt" --output-dir "$refined_dir" \
    2>&1 | tee "$run_root/refined.log"

echo "[$(date -Is)] Allen-Cahn paper-method extension R1 pipeline complete"

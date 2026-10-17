#!/usr/bin/env bash
set -euo pipefail

gpu=${1:-5}
project_dir=/home/whn/hpy_pinns
python_bin=/home/whn/miniconda3/envs/nan717/bin/python
result_root="$project_dir/results/table1_hpvpinn_burgers_paper_r1_20260810"
run_root="$project_dir/runs/table1_hpvpinn_burgers_paper_r1_20260810"
base_dir="$result_root/base_single_domain_p20q80"
refined_dir="$result_root/spatial_refined_p20q80"
mesh_file="$result_root/residual_spatial_strip.npy"
report_file="$result_root/spatial_refinement_report.json"
mkdir -p "$result_root" "$run_root"

common_args=(
    --problem burgers_1d
    --seed 1234
    --device cuda
    --dtype float64
    --variational-form r1
    --test-basis chebyshev_full
    --quadrature-rule clenshaw_curtis
    --n-test 20
    --n-quad 80
    --boundary-weight 1
    --initial-weight 1
    --checkpoint-every 2000
    --log-every 100
)

echo "[$(date -Is)] Burgers paper R1 base stage on physical GPU $gpu"
CUDA_VISIBLE_DEVICES="$gpu" "$python_bin" -u "$project_dir/hpvpinn_experiments.py" \
    "${common_args[@]}" --mesh-variant paper --iterations 36000 --output-dir "$base_dir" \
    2>&1 | tee "$run_root/base.log"

echo "[$(date -Is)] Burgers residual-driven spatial strip"
CUDA_VISIBLE_DEVICES="$gpu" "$python_bin" -u "$project_dir/refine_burgers_spatial_strip.py" \
    "$base_dir" --device cuda --theta 0.5 --x-order 201 --t-order 101 \
    --output-mesh "$mesh_file" --output-report "$report_file" \
    2>&1 | tee "$run_root/refine.log"

echo "[$(date -Is)] Burgers paper R1 spatial-refined stage"
CUDA_VISIBLE_DEVICES="$gpu" "$python_bin" -u "$project_dir/hpvpinn_experiments.py" \
    "${common_args[@]}" --elements-file "$mesh_file" --iterations 24000 \
    --init-checkpoint "$base_dir/checkpoint_latest.pt" --output-dir "$refined_dir" \
    2>&1 | tee "$run_root/refined.log"

echo "[$(date -Is)] Burgers paper R1 pipeline complete"

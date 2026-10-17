#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 4 ]]; then
    echo "usage: $0 {burgers_1d|allen_cahn_1d} GPU RESULT_ROOT RUN_ROOT" >&2
    exit 2
fi

problem=$1
gpu=$2
result_root=$3
run_root=$4
project_dir=/home/whn/hpy_pinns
python_bin=/home/whn/miniconda3/envs/nan717/bin/python

case "$problem" in
    burgers_1d)
        base_iterations=36000
        continuation_iterations=12000
        initial_weight=1
        extra_args=()
        ;;
    allen_cahn_1d)
        base_iterations=180000
        continuation_iterations=60000
        initial_weight=100
        extra_args=(--fourier-modes 1)
        ;;
    *)
        echo "unsupported problem: $problem" >&2
        exit 2
        ;;
esac

problem_results="$result_root/$problem/seed_1234"
problem_runs="$run_root/$problem"
base_dir="$problem_results/base_p5q10"
h_dir="$problem_results/h_refined_p5q10"
hp_dir="$problem_results/hp_refined_p8q16"
mesh_file="$problem_results/residual_refined_elements.npy"
report_file="$problem_results/refinement_report.json"
mkdir -p "$problem_runs" "$problem_results"

common_args=(
    --problem "$problem"
    --seed 1234
    --device cuda
    --dtype float64
    --variational-form p1
    --temporal-test-basis bubble
    --boundary-weight 1
    --initial-weight "$initial_weight"
    --checkpoint-every 2000
    --log-every 100
)

echo "[$(date -Is)] $problem base stage on physical GPU $gpu"
CUDA_VISIBLE_DEVICES="$gpu" "$python_bin" -u "$project_dir/hpvpinn_experiments.py" \
    "${common_args[@]}" "${extra_args[@]}" \
    --iterations "$base_iterations" --n-test 5 --n-quad 10 --output-dir "$base_dir" \
    2>&1 | tee "$problem_runs/base.log"

echo "[$(date -Is)] $problem residual indicator and h-refinement"
CUDA_VISIBLE_DEVICES="$gpu" "$python_bin" -u "$project_dir/refine_hpvpinn_mesh.py" \
    "$base_dir" --device cuda --theta 0.5 --quadrature-order 20 \
    --output-mesh "$mesh_file" --output-report "$report_file" \
    2>&1 | tee "$problem_runs/refine.log"

echo "[$(date -Is)] $problem h stage"
CUDA_VISIBLE_DEVICES="$gpu" "$python_bin" -u "$project_dir/hpvpinn_experiments.py" \
    "${common_args[@]}" "${extra_args[@]}" \
    --iterations "$continuation_iterations" --n-test 5 --n-quad 10 \
    --elements-file "$mesh_file" --init-checkpoint "$base_dir/checkpoint_latest.pt" \
    --output-dir "$h_dir" 2>&1 | tee "$problem_runs/h_refined.log"

echo "[$(date -Is)] $problem p stage"
CUDA_VISIBLE_DEVICES="$gpu" "$python_bin" -u "$project_dir/hpvpinn_experiments.py" \
    "${common_args[@]}" "${extra_args[@]}" \
    --iterations "$continuation_iterations" --n-test 8 --n-quad 16 \
    --elements-file "$mesh_file" --init-checkpoint "$h_dir/checkpoint_latest.pt" \
    --output-dir "$hp_dir" 2>&1 | tee "$problem_runs/hp_refined.log"

echo "[$(date -Is)] $problem adaptive pipeline complete"

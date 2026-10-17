#!/usr/bin/env bash
set -euo pipefail

gpu=${1:-7}
project_dir=/home/whn/hpy_pinns
python_bin=/home/whn/miniconda3/envs/nan717/bin/python
result_root="$project_dir/results/table1_hpvpinn_r1_corrected_20260810"
run_root="$project_dir/runs/table1_hpvpinn_r1_corrected_20260810"

burgers_dir="$result_root/burgers_1d/paper_figure_fine_p20q80_seed_1234"
allen_dir="$result_root/allen_cahn_1d/causal_h8x16_p5q10_seed_1234"
mkdir -p "$burgers_dir" "$allen_dir" "$run_root"

CUDA_VISIBLE_DEVICES="$gpu" "$python_bin" -u "$project_dir/hpvpinn_experiments.py" \
    --problem burgers_1d \
    --seed 1234 \
    --device cuda \
    --dtype float64 \
    --variational-form r1 \
    --test-basis chebyshev_full \
    --quadrature-rule clenshaw_curtis \
    --mesh-variant paper_figure_fine \
    --n-test 20 \
    --n-quad 80 \
    --boundary-weight 1 \
    --initial-weight 1 \
    --learning-rate 0.001 \
    --lr-step 2000 \
    --lr-gamma 0.9 \
    --iterations 60000 \
    --checkpoint-every 2000 \
    --log-every 100 \
    --output-dir "$burgers_dir" \
    >"$run_root/burgers.log" 2>&1 &
burgers_pid=$!

CUDA_VISIBLE_DEVICES="$gpu" "$python_bin" -u "$project_dir/hpvpinn_experiments.py" \
    --problem allen_cahn_1d \
    --seed 1234 \
    --device cuda \
    --dtype float64 \
    --variational-form r1 \
    --test-basis chebyshev_full \
    --quadrature-rule clenshaw_curtis \
    --mesh-variant causal_refined \
    --n-test 5 \
    --n-quad 10 \
    --fourier-modes 1 \
    --boundary-weight 1 \
    --initial-weight 100 \
    --learning-rate 0.001 \
    --lr-step 2000 \
    --lr-gamma 0.9 \
    --temporal-curriculum-stages 16 \
    --curriculum-final-fraction 0.5 \
    --iterations 60000 \
    --checkpoint-every 2000 \
    --log-every 100 \
    --output-dir "$allen_dir" \
    >"$run_root/allen_cahn.log" 2>&1 &
allen_pid=$!

printf 'Burgers PID: %s\n' "$burgers_pid"
printf 'Allen-Cahn PID: %s\n' "$allen_pid"

status=0
wait "$burgers_pid" || status=1
wait "$allen_pid" || status=1
exit "$status"

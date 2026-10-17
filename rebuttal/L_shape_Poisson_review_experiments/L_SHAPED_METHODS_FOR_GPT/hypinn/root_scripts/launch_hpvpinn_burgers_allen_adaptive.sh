#!/usr/bin/env bash
set -euo pipefail

project_dir=/home/whn/hpy_pinns
result_root="$project_dir/results/table1_hpvpinn_paper_adaptive_20260810"
run_root="$project_dir/runs/table1_hpvpinn_paper_adaptive_20260810"
pipeline="$project_dir/run_hpvpinn_adaptive_pipeline.sh"

mkdir -p "$result_root" "$run_root"
tmux new-session -d -s hpv_burgers_adaptive \
    "$pipeline" burgers_1d 5 "$result_root" "$run_root"
tmux new-session -d -s hpv_allen_adaptive \
    "$pipeline" allen_cahn_1d 6 "$result_root" "$run_root"

echo "launched hpv_burgers_adaptive on GPU 5"
echo "launched hpv_allen_adaptive on GPU 6"

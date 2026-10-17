#!/usr/bin/env bash
set -euo pipefail

SESSION="sa_dasa_full_seed0"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SUITE_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PROJECT_ROOT="$(cd "$SUITE_ROOT/../.." && pwd)"
PYTHON="$PROJECT_ROOT/.conda-env/bin/python"
RUN_ROOT="${1:-$SUITE_ROOT/runs/seed0}"

if tmux has-session -t "$SESSION" 2>/dev/null; then
    echo "tmux session already exists: $SESSION" >&2
    exit 1
fi

mkdir -p "$RUN_ROOT"
tmux new-session -d -s "$SESSION" -n monitor \
    "cd '$PROJECT_ROOT' && '$PYTHON' -m benchmarks.dasa_aligned_sa.scripts.monitor_runs --root '$RUN_ROOT' --watch"

launch_case() {
    local case_id="$1"
    local gpu_id="$2"
    local output="$RUN_ROOT/$case_id"
    mkdir -p "$output"
    tmux new-window -t "$SESSION" -n "$case_id" \
        "bash -lc \"set -o pipefail; cd '$PROJECT_ROOT'; CUDA_VISIBLE_DEVICES=$gpu_id TF_FORCE_GPU_ALLOW_GROWTH=true '$PYTHON' -m benchmarks.dasa_aligned_sa.scripts.run_case --case '$case_id' --seed 0 --output '$output' 2>&1 | tee '$output/train.log'\""
}

launch_case allen_cahn 0
launch_case diffusion_reaction 1
launch_case advection_diffusion 2
launch_case cavity 3
launch_case poisson_1d_boundary 3
launch_case lshape_poisson 4
launch_case poisson_1d_steep 4
launch_case helmholtz_2d 5
launch_case helmholtz_1d 5
launch_case burgers 6
launch_case poisson_2d_peak 6
launch_case discontinuous_advection 7
launch_case poisson_2d_steep 7

tmux select-window -t "$SESSION:monitor"
echo "started tmux session $SESSION"
echo "attach with: tmux attach -t $SESSION"

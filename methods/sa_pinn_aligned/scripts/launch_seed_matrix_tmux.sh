#!/usr/bin/env bash
set -euo pipefail

SESSION="${SA_DASA_SESSION:-sa_dasa_seeds_0_9_v2}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SUITE_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PROJECT_ROOT="$(cd "$SUITE_ROOT/../.." && pwd)"
PYTHON="$PROJECT_ROOT/.conda-env/bin/python"

if [[ "${1:-}" == "--worker" ]]; then
    GPU_ID="$2"
    RUN_ROOT="$3"
    QUEUE="$RUN_ROOT/.queue"
    shopt -s nullglob

    while true; do
        claim=""
        for task in "$QUEUE"/*.pending; do
            running="${task%.pending}.running"
            if mv "$task" "$running" 2>/dev/null; then
                claim="$running"
                break
            fi
        done
        [[ -n "$claim" ]] || exit 0

        IFS=$'\t' read -r case_id seed < "$claim"
        output="$RUN_ROOT/seed_$seed/$case_id"
        mkdir -p "$output"
        set +e
        CUDA_VISIBLE_DEVICES="$GPU_ID" TF_FORCE_GPU_ALLOW_GROWTH=true \
            "$PYTHON" -m methods.sa_pinn_aligned.scripts.run_case \
            --case "$case_id" --seed "$seed" --output "$output" \
            2>&1 | tee "$output/train.log"
        status="${PIPESTATUS[0]}"
        set -e
        if [[ "$status" -eq 0 ]]; then
            mv "$claim" "${claim%.running}.done"
        else
            mv "$claim" "${claim%.running}.failed"
        fi
    done
fi

RUN_ROOT="${1:-$SUITE_ROOT/runs/seeds_0_9_v2}"
GPU_IDS=("${@:2}")
if [[ "${#GPU_IDS[@]}" -eq 0 ]]; then
    GPU_IDS=(0 1 2 3 4 5 6 7)
fi

if tmux has-session -t "$SESSION" 2>/dev/null; then
    echo "tmux session already exists: $SESSION" >&2
    exit 1
fi
if [[ -e "$RUN_ROOT" ]]; then
    echo "run output already exists: $RUN_ROOT" >&2
    exit 1
fi

mkdir -p "$RUN_ROOT/.queue"
task_index=0
cases=(
    allen_cahn
    advection_diffusion
    diffusion_reaction
    discontinuous_advection
    burgers
    cavity
    lshape_poisson
    helmholtz_2d
    helmholtz_1d
    poisson_1d_boundary
    poisson_1d_steep
    poisson_2d_peak
    poisson_2d_steep
)
seeds=(0 1 2 3 4 5 6 7 8 9)
if [[ -n "${SA_DASA_CASES:-}" ]]; then
    read -r -a cases <<< "$SA_DASA_CASES"
fi
if [[ -n "${SA_DASA_SEEDS:-}" ]]; then
    read -r -a seeds <<< "$SA_DASA_SEEDS"
fi
for case_id in "${cases[@]}"; do
    for seed in "${seeds[@]}"; do
        task_name=$(printf 'task_%03d_%s_seed_%02d' "$task_index" "$case_id" "$seed")
        printf '%s\t%s\n' "$case_id" "$seed" > "$RUN_ROOT/.queue/$task_name.pending"
        task_index=$((task_index + 1))
    done
done

tmux new-session -d -s "$SESSION" -n monitor \
    "watch -n 30 \"printf 'complete '; find '$RUN_ROOT' -name metrics.json | wc -l; printf 'failed '; find '$RUN_ROOT' -name failure.json | wc -l; printf 'queued '; find '$RUN_ROOT/.queue' -name '*.pending' | wc -l; printf 'running '; find '$RUN_ROOT/.queue' -name '*.running' | wc -l\""

for gpu_id in "${GPU_IDS[@]}"; do
    output_log="$RUN_ROOT/worker_gpu${gpu_id}.log"
    tmux new-window -t "$SESSION" -n "gpu${gpu_id}" \
        "bash '$SCRIPT_DIR/launch_seed_matrix_tmux.sh' --worker '$gpu_id' '$RUN_ROOT' 2>&1 | tee '$output_log'"
done

tmux select-window -t "$SESSION:monitor"
echo "started tmux session $SESSION"
echo "queued tasks: $task_index"
echo "GPUs: ${GPU_IDS[*]}"
echo "attach with: tmux attach -t $SESSION"

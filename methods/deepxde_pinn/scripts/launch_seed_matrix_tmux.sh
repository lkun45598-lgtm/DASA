#!/usr/bin/env bash
set -euo pipefail

SESSION="${DEEPXDE_PINN_SESSION:-deepxde_pinn_seeds_0_9}"
TRAINING_PROFILE="${DEEPXDE_PINN_TRAINING_PROFILE:-case_config}"
WORKERS_PER_GPU="${DEEPXDE_PINN_WORKERS_PER_GPU:-1}"
LBFGS_STEPS="${DEEPXDE_PINN_LBFGS_STEPS:-}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SUITE_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PROJECT_ROOT="$(cd "$SUITE_ROOT/../.." && pwd)"
PYTHON="${DEEPXDE_PINN_PYTHON:-/home/lz/miniconda3/envs/pytorch/bin/python}"

if [[ "${1:-}" == "--worker" ]]; then
    GPU_ID="$2"
    RUN_ROOT="$3"
    PROFILE="${4:-$TRAINING_PROFILE}"
    LBFGS_OVERRIDE="${5:-$LBFGS_STEPS}"
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
        step_args=()
        if [[ -n "$LBFGS_OVERRIDE" ]]; then
            step_args=(--lbfgs-steps "$LBFGS_OVERRIDE")
        fi
        set +e
        DDE_BACKEND=pytorch MPLCONFIGDIR=/tmp/matplotlib-deepxde \
            CUDA_VISIBLE_DEVICES="$GPU_ID" CUBLAS_WORKSPACE_CONFIG=:4096:8 \
            PYTHONUNBUFFERED=1 "$PYTHON" -m methods.deepxde_pinn.scripts.run_case \
            --case "$case_id" --seed "$seed" --output "$output" \
            --training-profile "$PROFILE" \
            "${step_args[@]}" \
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

if [[ "${1:-}" == "--finalizer" ]]; then
    RUN_ROOT="$2"
    while find "$RUN_ROOT/.queue" -type f \( -name '*.pending' -o -name '*.running' \) -print -quit | grep -q .; do
        sleep 30
    done
    DDE_BACKEND=pytorch MPLCONFIGDIR=/tmp/matplotlib-deepxde \
        "$PYTHON" -m methods.deepxde_pinn.scripts.aggregate_results --root "$RUN_ROOT"
    exit 0
fi

PREPARE_ONLY=false
if [[ "${1:-}" == "--prepare-only" ]]; then
    PREPARE_ONLY=true
    shift
fi

RESUME=false
if [[ "${1:-}" == "--resume" ]]; then
    RESUME=true
    shift
fi
RUN_ROOT="${1:-$SUITE_ROOT/runs/seeds_0_9}"
if [[ "$#" -gt 0 ]]; then
    shift
fi
GPU_IDS=("$@")
if [[ "${#GPU_IDS[@]}" -eq 0 ]]; then
    GPU_IDS=(0 1 2 3 4 5 6 7)
fi

if [[ "$PREPARE_ONLY" == false ]] && tmux has-session -t "$SESSION" 2>/dev/null; then
    echo "tmux session already exists: $SESSION" >&2
    exit 1
fi

cases=(
    advection_diffusion
    allen_cahn
    burgers
    cavity
    diffusion_reaction
    discontinuous_advection
    helmholtz_1d
    helmholtz_2d
    lshape_poisson
    poisson_1d_boundary
    poisson_1d_steep
    poisson_2d_peak
    poisson_2d_steep
)
seeds=(0 1 2 3 4 5 6 7 8 9)
if [[ -n "${DEEPXDE_PINN_CASES:-}" ]]; then
    read -r -a cases <<< "$DEEPXDE_PINN_CASES"
fi
if [[ -n "${DEEPXDE_PINN_SEEDS:-}" ]]; then
    read -r -a seeds <<< "$DEEPXDE_PINN_SEEDS"
fi
if [[ "${#cases[@]}" -eq 0 || "${#seeds[@]}" -eq 0 ]]; then
    echo "case and seed selections must be non-empty" >&2
    exit 1
fi
if [[ ! "$WORKERS_PER_GPU" =~ ^[1-9][0-9]*$ ]]; then
    echo "DEEPXDE_PINN_WORKERS_PER_GPU must be a positive integer" >&2
    exit 1
fi
if [[ -n "$LBFGS_STEPS" && ! "$LBFGS_STEPS" =~ ^[0-9]+$ ]]; then
    echo "DEEPXDE_PINN_LBFGS_STEPS must be a nonnegative integer" >&2
    exit 1
fi

if [[ "$RESUME" == true ]]; then
    if [[ ! -d "$RUN_ROOT/.queue" ]]; then
        echo "missing resumable queue: $RUN_ROOT/.queue" >&2
        exit 1
    fi
    shopt -s nullglob
    for task in "$RUN_ROOT/.queue"/*.running; do
        mv "$task" "${task%.running}.pending"
    done
else
    if [[ -e "$RUN_ROOT" ]]; then
        echo "run output already exists: $RUN_ROOT" >&2
        exit 1
    fi
    mkdir -p "$RUN_ROOT/.queue"
    task_index=0
    for case_id in "${cases[@]}"; do
        for seed in "${seeds[@]}"; do
            task_name=$(printf 'task_%03d_%s_seed_%02d' "$task_index" "$case_id" "$seed")
            printf '%s\t%s\n' "$case_id" "$seed" > "$RUN_ROOT/.queue/$task_name.pending"
            task_index=$((task_index + 1))
        done
    done
fi

{
    date -u
    nvidia-smi || true
    DDE_BACKEND=pytorch MPLCONFIGDIR=/tmp/matplotlib-deepxde "$PYTHON" -c \
        'import deepxde, torch; print("deepxde", deepxde.__version__); print("torch", torch.__version__); print("cuda", torch.version.cuda)'
} > "$RUN_ROOT/environment.txt" 2>&1

{
    echo "training_profile=$TRAINING_PROFILE"
    echo "cases=${cases[*]}"
    echo "seeds=${seeds[*]}"
    echo "workers_per_gpu=$WORKERS_PER_GPU"
    echo "lbfgs_steps=${LBFGS_STEPS:-profile_default}"
} > "$RUN_ROOT/queue_manifest.txt"

if [[ "$PREPARE_ONLY" == true ]]; then
    echo "prepared $RUN_ROOT"
    exit 0
fi

tmux new-session -d -s "$SESSION" -n monitor \
    "watch -n 30 \"printf 'complete '; find '$RUN_ROOT' -name metrics.json | wc -l; printf 'failed '; find '$RUN_ROOT/.queue' -name '*.failed' | wc -l; printf 'pending '; find '$RUN_ROOT/.queue' -name '*.pending' | wc -l; printf 'running '; find '$RUN_ROOT/.queue' -name '*.running' | wc -l\""

for gpu_id in "${GPU_IDS[@]}"; do
    for ((worker_id = 0; worker_id < WORKERS_PER_GPU; worker_id++)); do
        tmux new-window -t "$SESSION" -n "g${gpu_id}w${worker_id}" \
            "bash '$SCRIPT_DIR/launch_seed_matrix_tmux.sh' --worker '$gpu_id' '$RUN_ROOT' '$TRAINING_PROFILE' '$LBFGS_STEPS' 2>&1 | tee '$RUN_ROOT/worker_gpu${gpu_id}_${worker_id}.log'"
    done
done
tmux new-window -t "$SESSION" -n finalizer \
    "bash '$SCRIPT_DIR/launch_seed_matrix_tmux.sh' --finalizer '$RUN_ROOT' 2>&1 | tee '$RUN_ROOT/finalizer.log'"
tmux select-window -t "$SESSION:monitor"

echo "started tmux session $SESSION"
echo "run root: $RUN_ROOT"
echo "GPUs: ${GPU_IDS[*]}"
echo "workers per GPU: $WORKERS_PER_GPU"
echo "training profile: $TRAINING_PROFILE"
echo "L-BFGS steps: ${LBFGS_STEPS:-profile default}"
echo "attach with: tmux attach -t $SESSION"

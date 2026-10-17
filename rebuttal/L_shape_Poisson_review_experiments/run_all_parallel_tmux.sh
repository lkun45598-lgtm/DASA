#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT="$ROOT/$(basename "${BASH_SOURCE[0]}")"
PYTHON="${PYTHON:-/home/lz/miniconda3/envs/pytorch/bin/python}"
SESSION="${SESSION:-lshape_rebuttal_formal}"
SEEDS=(0 1 2)

usage() {
    cat <<EOF
Usage:
  bash $(basename "$0") [RUN_ROOT] [GPU_ID ...]
  bash $(basename "$0") --resume RUN_ROOT [GPU_ID ...]
  bash $(basename "$0") --prepare-only RUN_ROOT

Environment:
  PYTHON   Python executable (default: $PYTHON)
  SESSION  tmux session name (default: $SESSION)
EOF
}

count_files() {
    find "$1" -maxdepth 1 -type f -name "$2" -print | wc -l | tr -d ' '
}

prepare_queue() {
    local run_root="$1"
    local queue="$run_root/.queue"
    local task_index=0
    local seed config task_name

    if [[ -e "$run_root" ]]; then
        echo "Run output already exists: $run_root" >&2
        echo "Use --resume to continue an interrupted run." >&2
        return 1
    fi
    mkdir -p "$queue"

    # Keep PINN and DASA for the same seed on one physical GPU for fair timing.
    for seed in "${SEEDS[@]}"; do
        task_name=$(printf 'task_%03d_cost_pair_seed_%02d' "$task_index" "$seed")
        {
            printf 'cost\t%s\t%s\n' "$ROOT/configs/cost/cost_pinn.json" "$seed"
            printf 'cost\t%s\t%s\n' "$ROOT/configs/cost/cost_dasa.json" "$seed"
        } > "$queue/$task_name.pending"
        task_index=$((task_index + 1))
    done

    for config in "$ROOT"/configs/sensitivity/*.json; do
        for seed in "${SEEDS[@]}"; do
            task_name=$(printf 'task_%03d_%s_seed_%02d' \
                "$task_index" "$(basename "${config%.json}")" "$seed")
            printf 'sensitivity\t%s\t%s\n' "$config" "$seed" \
                > "$queue/$task_name.pending"
            task_index=$((task_index + 1))
        done
    done

    echo "Prepared 30 queue units containing 33 training runs in $queue"
}

run_worker() {
    local gpu_id="$1"
    local run_root="$2"
    local queue="$run_root/.queue"
    local claim running group config seed output_root experiment output_dir
    local status unit_status
    local -a extra

    shopt -s nullglob
    while true; do
        claim=""
        for running in "$queue"/*.pending; do
            if mv "$running" "${running%.pending}.running" 2>/dev/null; then
                claim="${running%.pending}.running"
                break
            fi
        done
        [[ -n "$claim" ]] || return 0

        unit_status=0
        while IFS=$'\t' read -r group config seed; do
            if [[ "$group" == "cost" ]]; then
                output_root="$run_root/results_cost_adam"
                extra=(--lbfgs-outer-steps 0)
            elif [[ "$group" == "sensitivity" ]]; then
                output_root="$run_root/results_sensitivity_paper"
                extra=()
            else
                echo "Unknown task group '$group' in $claim" >&2
                unit_status=1
                continue
            fi

            experiment="$(basename "${config%.json}")"
            output_dir="$output_root/$experiment/seed_$seed"
            mkdir -p "$output_dir"
            echo "[$(date -Is)] GPU $gpu_id: $experiment seed $seed" \
                | tee -a "$output_dir/train.log"

            set +e
            CUDA_VISIBLE_DEVICES="$gpu_id" \
                CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}" \
                PYTHONUNBUFFERED=1 \
                "$PYTHON" "$ROOT/src/lshape_experiment.py" \
                --config "$config" \
                --seed "$seed" \
                --device cuda:0 \
                --output-root "$output_root" \
                "${extra[@]}" \
                2>&1 | tee -a "$output_dir/train.log"
            status="${PIPESTATUS[0]}"
            set -e
            if [[ "$status" -ne 0 ]]; then
                echo "[$(date -Is)] FAILED ($status): $experiment seed $seed" \
                    | tee -a "$output_dir/train.log"
                unit_status=1
            fi
        done < "$claim"

        if [[ "$unit_status" -eq 0 ]]; then
            mv "$claim" "${claim%.running}.done"
        else
            mv "$claim" "${claim%.running}.failed"
        fi
    done
}

show_status() {
    local run_root="$1"
    local queue="$run_root/.queue"
    local complete failed queued running

    complete=$(find "$run_root" -type f -name metrics.json -print | wc -l | tr -d ' ')
    failed=$(count_files "$queue" '*.failed')
    queued=$(count_files "$queue" '*.pending')
    running=$(count_files "$queue" '*.running')
    printf 'training runs: %s/33 complete\nqueue units: %s pending, %s running, %s failed\n' \
        "$complete" "$queued" "$running" "$failed"
}

finalize_results() {
    local run_root="$1"
    local queue="$run_root/.queue"
    local pending running failed

    while true; do
        pending=$(count_files "$queue" '*.pending')
        running=$(count_files "$queue" '*.running')
        [[ "$pending" -eq 0 && "$running" -eq 0 ]] && break
        sleep 30
    done

    "$PYTHON" "$ROOT/src/aggregate_results.py" \
        --results "$run_root/results_cost_adam" \
        --out "$run_root/results_cost_adam/summaries"
    "$PYTHON" "$ROOT/src/aggregate_results.py" \
        --results "$run_root/results_sensitivity_paper" \
        --out "$run_root/results_sensitivity_paper/summaries"

    failed=$(count_files "$queue" '*.failed')
    if [[ "$failed" -eq 0 ]]; then
        touch "$run_root/COMPLETE"
        echo "All 33 training runs completed."
    else
        printf '%s queue units failed.\n' "$failed" | tee "$run_root/FAILED"
        return 1
    fi
}

resume_queue() {
    local run_root="$1"
    local queue="$run_root/.queue"
    local task

    if [[ ! -d "$queue" ]]; then
        echo "Queue not found: $queue" >&2
        return 1
    fi
    shopt -s nullglob
    for task in "$queue"/*.running "$queue"/*.failed; do
        mv "$task" "${task%.*}.pending"
    done
}

launch_tmux() {
    local run_root="$1"
    shift
    local -a gpu_ids=("$@")
    local gpu_id worker_log monitor_command worker_command finalizer_command

    if [[ "${#gpu_ids[@]}" -eq 0 ]]; then
        gpu_ids=(0 1 2 3 4 5 6 7)
    fi
    if [[ ! -x "$PYTHON" ]]; then
        echo "Python executable is unavailable: $PYTHON" >&2
        return 1
    fi
    if ! command -v tmux >/dev/null 2>&1; then
        echo "tmux is not installed." >&2
        return 1
    fi
    if tmux has-session -t "$SESSION" 2>/dev/null; then
        echo "tmux session already exists: $SESSION" >&2
        return 1
    fi
    for gpu_id in "${gpu_ids[@]}"; do
        [[ "$gpu_id" =~ ^[0-9]+$ ]] || {
            echo "Invalid GPU ID: $gpu_id" >&2
            return 1
        }
    done

    printf -v monitor_command 'watch -n 30 bash\ %q\ --status\ %q' "$SCRIPT" "$run_root"
    tmux new-session -d -s "$SESSION" -n monitor "$monitor_command"

    for gpu_id in "${gpu_ids[@]}"; do
        worker_log="$run_root/worker_gpu${gpu_id}.log"
        printf -v worker_command 'bash %q --worker %q %q 2>&1 | tee -a %q' \
            "$SCRIPT" "$gpu_id" "$run_root" "$worker_log"
        tmux new-window -t "$SESSION" -n "gpu${gpu_id}" "$worker_command"
    done

    printf -v finalizer_command 'bash %q --finalize %q 2>&1 | tee -a %q' \
        "$SCRIPT" "$run_root" "$run_root/finalize.log"
    tmux new-window -t "$SESSION" -n finalize "$finalizer_command"
    tmux select-window -t "$SESSION:monitor"

    echo "Started tmux session: $SESSION"
    echo "Run root: $run_root"
    echo "GPUs: ${gpu_ids[*]}"
    echo "Attach with: tmux attach -t $SESSION"
}

case "${1:-}" in
    --help|-h)
        usage
        ;;
    --prepare-only)
        [[ "$#" -eq 2 ]] || { usage >&2; exit 2; }
        prepare_queue "$2"
        ;;
    --worker)
        [[ "$#" -eq 3 ]] || { usage >&2; exit 2; }
        run_worker "$2" "$3"
        ;;
    --status)
        [[ "$#" -eq 2 ]] || { usage >&2; exit 2; }
        show_status "$2"
        ;;
    --finalize)
        [[ "$#" -eq 2 ]] || { usage >&2; exit 2; }
        finalize_results "$2"
        ;;
    --resume)
        [[ "$#" -ge 2 ]] || { usage >&2; exit 2; }
        run_root="$2"
        shift 2
        resume_queue "$run_root"
        launch_tmux "$run_root" "$@"
        ;;
    --*)
        echo "Unknown option: $1" >&2
        usage >&2
        exit 2
        ;;
    *)
        run_root="${1:-$ROOT/results_rebuttal_formal}"
        [[ "$#" -eq 0 ]] || shift
        prepare_queue "$run_root"
        launch_tmux "$run_root" "$@"
        ;;
esac

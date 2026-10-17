#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT="$ROOT/$(basename "${BASH_SOURCE[0]}")"
RUNNER="$ROOT/wave_experiment.py"
AGGREGATOR="$ROOT/aggregate_results.py"
PYTHON="${PYTHON:-/home/lz/miniconda3/envs/pytorch/bin/python}"
SESSION="${SESSION:-wave_rebuttal_formal}"
TOTAL_RUNS=123
SCOPE="${SCOPE:-all}"
SHARD_INDEX="${SHARD_INDEX:-0}"
SHARD_COUNT="${SHARD_COUNT:-1}"

queue_count() {
    local queue="$1" pattern="$2"
    find "$queue" -maxdepth 1 -type f -name "$pattern" -print 2>/dev/null | wc -l | tr -d ' '
}

init_queue() {
    local results="$1"
    local queue="$results/.queue" index=0 seed variant task total_runs=0 stage_index=0
    if [[ -e "$results" ]]; then
        echo "Results path already exists: $results" >&2
        return 1
    fi
    [[ "$SCOPE" == "all" || "$SCOPE" == "main" || "$SCOPE" == "stage" ]] || {
        echo "SCOPE must be all, main, or stage" >&2
        return 1
    }
    [[ "$SHARD_INDEX" =~ ^[0-9]+$ && "$SHARD_COUNT" =~ ^[1-9][0-9]*$ ]] || {
        echo "SHARD_INDEX and SHARD_COUNT must be non-negative/positive integers" >&2
        return 1
    }
    [[ "$SHARD_INDEX" -lt "$SHARD_COUNT" ]] || {
        echo "SHARD_INDEX must be smaller than SHARD_COUNT" >&2
        return 1
    }
    mkdir -p "$queue"

    # A PINN/DASA seed pair stays on one GPU so timing sees the same hardware load.
    if [[ "$SCOPE" == "all" || "$SCOPE" == "main" ]]; then
        for seed in 0 1 2; do
            task=$(printf '%s/task_%03d_cost_seed_%02d.pending' "$queue" "$index" "$seed")
            printf 'cost\tpinn\t%s\ncost\tdasa\t%s\n' "$seed" "$seed" > "$task"
            index=$((index + 1))
            total_runs=$((total_runs + 2))
        done

        for variant in \
            baseline arch_3x5 arch_7x20 lr_both_5e-4 lr_subnet_5e-4 \
            lr_subnet_2e-3 lr_both_2e-3 ratio_1to5 ratio_5to1; do
            for seed in 0 1 2; do
                task=$(printf '%s/task_%03d_sensitivity_%s_seed_%02d.pending' \
                    "$queue" "$index" "$variant" "$seed")
                printf 'sensitivity\t%s\t%s\n' "$variant" "$seed" > "$task"
                index=$((index + 1))
                total_runs=$((total_runs + 1))
            done
        done
    fi

    if [[ "$SCOPE" == "all" || "$SCOPE" == "stage" ]]; then
        for variant in \
            baseline stage1_2500 stage1_10000 stage2_1000 stage2_4000 \
            stage3_25000 stage3_100000 stage4_25000 stage4_100000; do
            for seed in 0 1 2 3 4 5 6 7 8 9; do
                if [[ $((stage_index % SHARD_COUNT)) -eq "$SHARD_INDEX" ]]; then
                    task=$(printf '%s/task_%03d_stage_length_%s_seed_%02d.pending' \
                        "$queue" "$index" "$variant" "$seed")
                    printf 'stage_length\t%s\t%s\n' "$variant" "$seed" > "$task"
                    index=$((index + 1))
                    total_runs=$((total_runs + 1))
                fi
                stage_index=$((stage_index + 1))
            done
        done
    fi
    printf '%s\n' "$total_runs" > "$queue/total_runs"
    printf '%s\n' "$SCOPE" > "$queue/scope"
    printf '%s/%s\n' "$SHARD_INDEX" "$SHARD_COUNT" > "$queue/shard"
    echo "Prepared $index queue units containing $total_runs runs (scope=$SCOPE shard=$SHARD_INDEX/$SHARD_COUNT)."
}

worker() {
    local gpu="$1" results="$2" worker_id="$3"
    local queue="$results/.queue"
    local pending claim group variant seed run_dir status unit_status
    shopt -s nullglob
    while true; do
        claim=""
        for pending in "$queue"/*.pending; do
            if mv "$pending" "${pending%.pending}.running" 2>/dev/null; then
                claim="${pending%.pending}.running"
                break
            fi
        done
        [[ -n "$claim" ]] || break

        unit_status=0
        while IFS=$'\t' read -r group variant seed; do
            run_dir="$results/$group/${group}_${variant}/seed_$seed"
            if [[ -f "$run_dir/metrics.json" ]]; then
                continue
            fi
            mkdir -p "$run_dir"
            echo "[$(date -Is)] worker=$worker_id gpu=$gpu group=$group variant=$variant seed=$seed" \
                | tee -a "$run_dir/train.log"
            set +e
            CUDA_VISIBLE_DEVICES="$gpu" \
                CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}" \
                PYTHONUNBUFFERED=1 \
                "$PYTHON" "$RUNNER" \
                    --group "$group" \
                    --variant "$variant" \
                    --seed "$seed" \
                    --device cuda:0 \
                    --output-root "$results" \
                    >> "$run_dir/train.log" 2>&1
            status=$?
            set -e
            if [[ "$status" -ne 0 || ! -f "$run_dir/metrics.json" ]]; then
                echo "[$(date -Is)] FAILED status=$status" | tee -a "$run_dir/train.log"
                unit_status=1
                break
            fi
        done < "$claim"

        if [[ "$unit_status" -eq 0 ]]; then
            mv "$claim" "${claim%.running}.done"
        else
            mv "$claim" "${claim%.running}.failed"
        fi
    done
    echo "[$(date -Is)] worker=$worker_id gpu=$gpu finished"
}

status_queue() {
    local results="$1"
    local queue="$results/.queue" expected="$TOTAL_RUNS"
    [[ -f "$queue/total_runs" ]] && expected="$(< "$queue/total_runs")"
    printf 'metrics: %s/%s\n' \
        "$(find "$results" -type f -name metrics.json -not -path '*/.queue/*' -print 2>/dev/null | wc -l | tr -d ' ')" \
        "$expected"
    printf 'queue units: %s pending, %s running, %s failed, %s done\n' \
        "$(queue_count "$queue" '*.pending')" \
        "$(queue_count "$queue" '*.running')" \
        "$(queue_count "$queue" '*.failed')" \
        "$(queue_count "$queue" '*.done')"
}

resume_queue() {
    local results="$1"
    local queue="$results/.queue" task
    shopt -s nullglob
    for task in "$queue"/*.running "$queue"/*.failed; do
        mv "$task" "${task%.*}.pending"
    done
}

finalize() {
    local results="$1"
    local queue="$results/.queue" pending running failed expected="$TOTAL_RUNS"
    [[ -f "$queue/total_runs" ]] && expected="$(< "$queue/total_runs")"
    while true; do
        pending=$(queue_count "$queue" '*.pending')
        running=$(queue_count "$queue" '*.running')
        [[ "$pending" -eq 0 && "$running" -eq 0 ]] && break
        sleep 30
    done
    "$PYTHON" "$AGGREGATOR" --results "$results" --out "$results/summaries"
    failed=$(queue_count "$queue" '*.failed')
    if [[ "$failed" -eq 0 ]]; then
        date -Is > "$results/COMPLETE"
        echo "All $expected runs completed."
    else
        echo "$failed queue units failed; use the resume command." | tee "$results/FAILED"
        return 1
    fi
}

launch() {
    local results="$1"
    shift
    local -a gpus=("$@")
    local gpu index command log
    [[ "${#gpus[@]}" -gt 0 ]] || gpus=(0 1 2 3 4 5 6 7)
    [[ -x "$PYTHON" ]] || { echo "Python not executable: $PYTHON" >&2; return 1; }
    command -v tmux >/dev/null || { echo "tmux is unavailable" >&2; return 1; }
    tmux has-session -t "$SESSION" 2>/dev/null && {
        echo "tmux session already exists: $SESSION" >&2
        return 1
    }

    printf -v command 'watch -n 30 bash\ %q\ status\ %q' "$SCRIPT" "$results"
    tmux new-session -d -s "$SESSION" -n monitor "$command"
    index=0
    for gpu in "${gpus[@]}"; do
        log="$results/worker_${index}_gpu${gpu}.log"
        printf -v command 'bash %q worker %q %q %q 2>&1 | tee -a %q' \
            "$SCRIPT" "$gpu" "$results" "$index" "$log"
        tmux new-window -t "$SESSION" -n "w${index}_g${gpu}" "$command"
        index=$((index + 1))
    done
    printf -v command 'bash %q finalize %q 2>&1 | tee -a %q' \
        "$SCRIPT" "$results" "$results/finalize.log"
    tmux new-window -t "$SESSION" -n finalize "$command"
    tmux select-window -t "$SESSION:monitor"
    echo "Started tmux session $SESSION with ${#gpus[@]} workers."
    echo "Results: $results"
}

case "${1:-}" in
    init)
        [[ "$#" -eq 2 ]] || exit 2
        init_queue "$2"
        ;;
    worker)
        [[ "$#" -eq 4 ]] || exit 2
        worker "$2" "$3" "$4"
        ;;
    status)
        [[ "$#" -eq 2 ]] || exit 2
        status_queue "$2"
        ;;
    resume)
        [[ "$#" -ge 2 ]] || exit 2
        results="$2"
        shift 2
        resume_queue "$results"
        launch "$results" "$@"
        ;;
    finalize)
        [[ "$#" -eq 2 ]] || exit 2
        finalize "$2"
        ;;
    start)
        [[ "$#" -ge 2 ]] || exit 2
        results="$2"
        shift 2
        init_queue "$results"
        launch "$results" "$@"
        ;;
    aggregate)
        [[ "$#" -eq 2 ]] || exit 2
        "$PYTHON" "$AGGREGATOR" --results "$2" --out "$2/summaries"
        ;;
    *)
        echo "usage: $0 {start RESULTS [GPU ...]|resume RESULTS [GPU ...]|status RESULTS|aggregate RESULTS}" >&2
        exit 2
        ;;
esac

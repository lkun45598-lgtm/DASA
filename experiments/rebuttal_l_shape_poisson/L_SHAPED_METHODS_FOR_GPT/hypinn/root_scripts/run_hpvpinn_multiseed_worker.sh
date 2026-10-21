#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: $0 PHYSICAL_GPU QUEUE_DIR" >&2
  exit 2
fi

physical_gpu=$1
queue_dir=$2
project_dir=/home/whn/hpy_pinns
task_runner="$project_dir/run_hpvpinn_multiseed_task.sh"
pending_file="$queue_dir/pending.tsv"
running_file="$queue_dir/running.tsv"
completed_file="$queue_dir/completed.tsv"
failed_file="$queue_dir/failed.tsv"
lock_file="$queue_dir/queue.lock"
log_dir="$queue_dir/../task_logs"

mkdir -p "$queue_dir" "$log_dir"
touch "$pending_file" "$running_file" "$completed_file" "$failed_file" "$lock_file"

active_problem=
active_seed=

remove_running_record() {
  local temporary="$running_file.tmp.$$"
  awk -F '\t' -v problem="$active_problem" -v seed="$active_seed" \
    '!($1 == problem && $2 == seed)' "$running_file" > "$temporary"
  mv "$temporary" "$running_file"
}

requeue_active_task() {
  if [[ -z "$active_problem" ]]; then
    return
  fi
  exec 9>"$lock_file"
  flock -x 9
  remove_running_record
  printf '%s\t%s\n' "$active_problem" "$active_seed" >> "$pending_file"
  printf '%s\t%s\t%s\t%s\tworker_interrupted\n' \
    "$active_problem" "$active_seed" "$physical_gpu" "$(date -Is)" >> "$failed_file"
  flock -u 9
  exec 9>&-
  active_problem=
  active_seed=
}

stop_worker() {
  requeue_active_task
  exit 143
}

trap requeue_active_task EXIT
trap stop_worker INT TERM

claim_task() {
  local task_line temporary
  exec 9>"$lock_file"
  flock -x 9
  task_line=$(sed -n '1p' "$pending_file")
  if [[ -z "$task_line" ]]; then
    flock -u 9
    exec 9>&-
    return 1
  fi
  temporary="$pending_file.tmp.$$"
  sed '1d' "$pending_file" > "$temporary"
  mv "$temporary" "$pending_file"
  IFS=$'\t' read -r active_problem active_seed <<< "$task_line"
  printf '%s\t%s\t%s\t%s\t%s\n' \
    "$active_problem" "$active_seed" "$physical_gpu" "$$" "$(date -Is)" >> "$running_file"
  flock -u 9
  exec 9>&-
}

finish_task() {
  local status=$1
  local exit_code=$2
  exec 9>"$lock_file"
  flock -x 9
  remove_running_record
  printf '%s\t%s\t%s\t%s\t%s\n' \
    "$active_problem" "$active_seed" "$physical_gpu" "$(date -Is)" "$exit_code" \
    >> "$status"
  flock -u 9
  exec 9>&-
  active_problem=
  active_seed=
}

echo "[$(date -Is)] worker start physical_gpu=$physical_gpu pid=$$"
while claim_task; do
  task_log="$log_dir/${active_problem}_seed_${active_seed}.log"
  echo "[$(date -Is)] claimed problem=$active_problem seed=$active_seed log=$task_log"
  if "$task_runner" "$active_problem" "$active_seed" "$physical_gpu" > "$task_log" 2>&1; then
    finish_task "$completed_file" 0
  else
    task_status=$?
    echo "[$(date -Is)] failed problem=$active_problem seed=$active_seed status=$task_status"
    finish_task "$failed_file" "$task_status"
  fi
done
echo "[$(date -Is)] worker queue empty physical_gpu=$physical_gpu"
trap - EXIT INT TERM

#!/usr/bin/env bash
set -euo pipefail

project_dir=/home/whn/hpy_pinns
queue_dir="$project_dir/runs/table1_hpvpinn_multiseed_20260811/queue"
candidate_dir="$project_dir/results/table1_hpvpinn_diffusion_reaction_candidates_20260811/p1_prefined_p10q20_continuation"
pending_file="$queue_dir/pending.tsv"
running_file="$queue_dir/running.tsv"
completed_file="$queue_dir/completed.tsv"
failed_file="$queue_dir/failed.tsv"
lock_file="$queue_dir/queue.lock"
problem=diffusion_reaction_1d

if [[ ! -s "$candidate_dir/metrics.json" || ! -s "$candidate_dir/config.json" ]]; then
  echo "fixed seed-1234 p10/q20 endpoint is not complete: $candidate_dir" >&2
  exit 2
fi

mkdir -p "$queue_dir"
touch "$pending_file" "$running_file" "$completed_file" "$failed_file" "$lock_file"

exec 9>"$lock_file"
flock -x 9

temporary="$pending_file.tmp.$$"
: > "$temporary"
added=0
for seed in 1235 1236 1237 1238 1239 1240 1241 1242 1243; do
  if awk -F '\t' -v problem="$problem" -v seed="$seed" \
    '$1 == problem && $2 == seed { found = 1 } END { exit !found }' \
    "$pending_file" "$running_file" "$completed_file" "$failed_file"; then
    continue
  fi
  printf '%s\t%s\n' "$problem" "$seed" >> "$temporary"
  added=$((added + 1))
done
sed -n '1,$p' "$pending_file" >> "$temporary"
mv "$temporary" "$pending_file"
printf '%s\tseed_1234_fixed_endpoint\t%s\n' \
  "$(date -Is)" "$candidate_dir" > "$queue_dir/diffusion_reaction_protocol.tsv"

flock -u 9
exec 9>&-

echo "added=$added problem=$problem pending=$(wc -l < "$pending_file")"

#!/usr/bin/env bash
set -euo pipefail

project_dir=/home/whn/hpy_pinns
candidate_metrics="$project_dir/results/table1_hpvpinn_diffusion_reaction_candidates_20260811/p1_prefined_p10q20_continuation/metrics.json"
log_prefix="[$(date -Is)]"

echo "$log_prefix waiting for fixed diffusion-reaction endpoint"
while [[ ! -s "$candidate_metrics" ]]; do
  sleep 30
done

echo "[$(date -Is)] validating fixed endpoint and existing seed-1234 runs"
/home/whn/miniconda3/envs/nan717/bin/python \
  "$project_dir/summarize_hpvpinn_multiseed.py" --allow-incomplete

echo "[$(date -Is)] enqueueing diffusion-reaction seeds 1235-1243"
exec "$project_dir/enqueue_hpvpinn_diffusion_reaction_multiseed.sh"

#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
project_root="$(cd "$script_dir/.." && pwd)"
output_root="$script_dir/../../results/burgers_1d_fixed_10seeds"
python_bin="${PYTHON_BIN:-python}"
forwarded_args=("$@")

for ((argument_index = 0; argument_index < ${#forwarded_args[@]}; argument_index++)); do
  case "${forwarded_args[$argument_index]}" in
    --output-root)
      next_index=$((argument_index + 1))
      if ((next_index >= ${#forwarded_args[@]})); then
        echo "--output-root requires a path" >&2
        exit 2
      fi
      output_root="${forwarded_args[$next_index]}"
      ;;
    --output-root=*)
      output_root="${forwarded_args[$argument_index]#--output-root=}"
      ;;
  esac
done

if [[ "$output_root" != /* ]]; then
  output_root="$project_root/$output_root"
fi
launcher_log="$output_root/nohup_launcher.log"
pid_file="$output_root/nohup_launcher.pid"

mkdir -p "$output_root"
cd "$project_root"

nohup "$python_bin" -u "$script_dir/run_10seeds.py" \
  --output-root "$output_root" \
  "${forwarded_args[@]}" \
  >"$launcher_log" 2>&1 </dev/null &

launcher_pid=$!
echo "$launcher_pid" >"$pid_file"

echo "Burgers launcher started with nohup."
echo "PID: $launcher_pid"
echo "Log: $launcher_log"
echo "PID file: $pid_file"

#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROTOCOL="burgers_rar_transfer"
OUTPUT_ROOT=""
FORWARDED=("$@")

for argument in "${FORWARDED[@]}"; do
  if [[ "$argument" == "-h" || "$argument" == "--help" ]]; then
    "$SCRIPT_DIR/run_smoke_then_full_10seeds.sh" --help
    exit 0
  fi
done

for ((index=0; index<${#FORWARDED[@]}; index++)); do
  if [[ "${FORWARDED[$index]}" == "--output-root" ]]; then
    if ((index + 1 >= ${#FORWARDED[@]})); then
      echo "--output-root requires a path" >&2
      exit 2
    fi
    OUTPUT_ROOT="${FORWARDED[$((index + 1))]}"
  elif [[ "${FORWARDED[$index]}" == --output-root=* ]]; then
    OUTPUT_ROOT="${FORWARDED[$index]#--output-root=}"
  elif [[ "${FORWARDED[$index]}" == "--protocol" ]]; then
    if ((index + 1 >= ${#FORWARDED[@]})); then
      echo "--protocol requires a value" >&2
      exit 2
    fi
    PROTOCOL="${FORWARDED[$((index + 1))]}"
  elif [[ "${FORWARDED[$index]}" == --protocol=* ]]; then
    PROTOCOL="${FORWARDED[$index]#--protocol=}"
  fi
done
if [[ -z "$OUTPUT_ROOT" ]]; then
  OUTPUT_ROOT="$SCRIPT_DIR/../../results/poisson_1d_${PROTOCOL}_10seeds"
fi
if [[ "$OUTPUT_ROOT" != /* ]]; then
  OUTPUT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)/$OUTPUT_ROOT"
fi
mkdir -p "$OUTPUT_ROOT"

PID_FILE="$OUTPUT_ROOT/nohup_launcher.pid"
if [[ -s "$PID_FILE" ]]; then
  EXISTING_PID="$(tr -cd '0-9' < "$PID_FILE")"
  if [[ -n "$EXISTING_PID" ]] && kill -0 "$EXISTING_PID" 2>/dev/null; then
    echo "Pipeline already running: PID=$EXISTING_PID" >&2
    echo "Log: $OUTPUT_ROOT/nohup_launcher.log" >&2
    exit 1
  fi
fi

nohup "$SCRIPT_DIR/run_smoke_then_full_10seeds.sh" \
  "${FORWARDED[@]}" \
  > "$OUTPUT_ROOT/nohup_launcher.log" 2>&1 </dev/null &

PID="$!"
echo "$PID" > "$PID_FILE"
echo "Started gPINN+RAR pipeline: PID=$PID"
echo "Log: $OUTPUT_ROOT/nohup_launcher.log"

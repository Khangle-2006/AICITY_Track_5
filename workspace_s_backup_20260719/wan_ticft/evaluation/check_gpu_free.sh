#!/usr/bin/env bash
set -euo pipefail

gpu="${1:?Usage: check_gpu_free.sh GPU_ID [MIN_FREE_MIB]}"
minimum="${2:-36000}"

free="$(
  nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$gpu" \
    | tr -d ' '
)"
util="$(
  nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits -i "$gpu" \
    | tr -d ' '
)"

echo "GPU $gpu: free=${free} MiB, utilization=${util}%"
if (( free < minimum )); then
  echo "GPU $gpu does not have the required ${minimum} MiB free." >&2
  exit 1
fi

#!/usr/bin/env bash
set -euo pipefail

ROOT=/workspace/experiments/wan_ticft/wan22_wts100_s10
INTERVAL="${1:-60}"

while true; do
  clear || true
  date
  echo

  echo "Benchmark processes"
  ps -eo pid,ppid,etime,cmd \
    | grep -E 'run_wts100_wan_region_benchmark|run_wan22_ti2v(_parallel)?' \
    | grep -v grep || true
  echo

  total="$(find "$ROOT" \( -type f -o -type l \) -name '*.mp4' 2>/dev/null | wc -l)"
  unique="$(find "$ROOT" \( -type f -o -type l \) -name '*.mp4' -printf '%f\n' 2>/dev/null | sort -u | wc -l)"
  duplicates=$((total - unique))
  logs="$(find "$ROOT" -type f -path '*/logs/*.log' 2>/dev/null | wc -l)"
  oom="$(grep -Ril 'CUDA out of memory' "$ROOT"/shard_*/logs 2>/dev/null | wc -l || true)"
  failed="$(grep -RilE 'Traceback|Error|Exception|Killed' "$ROOT"/shard_*/logs 2>/dev/null | wc -l || true)"

  echo "Progress"
  printf '  MP4 total:       %s\n' "$total"
  printf '  Samples unique:  %s / 100\n' "$unique"
  printf '  Duplicate MP4:   %s\n' "$duplicates"
  printf '  Sample logs:     %s\n' "$logs"
  printf '  OOM logs:        %s\n' "$oom"
  printf '  Error logs:      %s\n' "$failed"
  echo

  echo "Per shard"
  for shard in "$ROOT"/shard_*; do
    [[ -d "$shard" ]] || continue
    count="$(find "$shard" \( -type f -o -type l \) -name '*.mp4' 2>/dev/null | wc -l)"
    printf '  %-30s %s MP4\n' "$(basename "$shard")" "$count"
  done
  echo

  echo "Latest runner progress"
  grep -hE '^\[[0-9]+/100\]|^status|^finished shard|^failed_samples|^matched samples|^recommended region routes' \
    "$ROOT"/shard_*.log 2>/dev/null | tail -12 || true
  echo

  echo "Newest sample log"
  newest="$(
    find "$ROOT" -type f -path '*/logs/*.log' -printf '%T@ %p\n' 2>/dev/null \
      | sort -nr | head -1 | cut -d' ' -f2- || true
  )"
  if [[ -n "$newest" ]]; then
    echo "  $newest"
    tail -5 "$newest" || true
  fi

  echo
  echo "Refreshing every ${INTERVAL}s. Press Ctrl+C to stop monitoring."
  sleep "$INTERVAL"
done

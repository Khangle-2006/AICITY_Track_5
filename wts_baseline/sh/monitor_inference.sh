#!/usr/bin/env bash

# Automatically navigate to repository root
cd "$(dirname "$0")/.."
mkdir -p logs

set -eu

OUTPUT_DIR="/workspace/wts_baseline/prediction_wan_sota"
MONITOR_LOG="/workspace/wts_baseline/monitor.log"
MANIFEST="/workspace/wts_baseline/data/manifests/test.jsonl"
TOTAL_SAMPLES=$(wc -l < "$MANIFEST")

echo "=== Monitor started at $(date) ===" >> "$MONITOR_LOG"
echo "Total samples to generate: $TOTAL_SAMPLES" >> "$MONITOR_LOG"

while true; do
  # Count outputs
  OUTPUT_COUNT=$(ls -d "$OUTPUT_DIR"/*/ 2>/dev/null | wc -l)
  VALID_COUNT=0
  LATEST_DIR=""
  LATEST_SIZE=""
  
  for d in "$OUTPUT_DIR"/*/; do
    [ -d "$d" ] || continue
    f=$(ls "$d"*.png 2>/dev/null | head -1)
    if [ -n "$f" ]; then
      s=$(stat -c%s "$f" 2>/dev/null)
      [ "$s" -gt 100000 ] && VALID_COUNT=$((VALID_COUNT + 1))
      LATEST_DIR=$(basename "$d")
      LATEST_SIZE=$s
    fi
  done
  
  RUNNING=$(ps aux | grep "infer_wan.py" | grep -v grep | wc -l)
  
  echo "$(date) | running=$RUNNING | dirs=$OUTPUT_COUNT valid=$VALID_COUNT/$TOTAL_SAMPLES | latest=$LATEST_DIR ${LATEST_SIZE}B" >> "$MONITOR_LOG"
  
  # Stop if process is gone AND we have all samples
  if [ "$RUNNING" -eq 0 ]; then
    if [ "$VALID_COUNT" -ge "$TOTAL_SAMPLES" ]; then
      echo "=== ALL DONE at $(date): $VALID_COUNT/$TOTAL_SAMPLES valid samples ===" >> "$MONITOR_LOG"
      exit 0
    else
      echo "=== PROCESSES DIED at $(date): only $VALID_COUNT/$TOTAL_SAMPLES done ===" >> "$MONITOR_LOG"
      exit 1
    fi
  fi
  
  sleep 300
done

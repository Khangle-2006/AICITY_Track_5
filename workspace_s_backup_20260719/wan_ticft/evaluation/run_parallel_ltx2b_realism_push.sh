#!/usr/bin/env bash
set -euo pipefail

cd /workspace

ROOT=${ROOT:-/workspace/experiments/wan_ticft}
LOG_DIR=${LOG_DIR:-$ROOT/parallel_ltx2b_realism_logs}
GPU_DEV40=${GPU_DEV40:-2}
GPU_LORA=${GPU_LORA:-1}
DEV_LIMIT=${DEV_LIMIT:-4}
DEV_CHUNK=${DEV_CHUNK:-24}
LORA_PER_PHASE=${LORA_PER_PHASE:-20}
LORA_STEPS_NOTE="config fixed at 300 steps"

mkdir -p "$LOG_DIR"

echo "=== Parallel LTX2B realism push ==="
echo "logs=$LOG_DIR"
echo "dev40 gpu=$GPU_DEV40 limit=$DEV_LIMIT chunk=$DEV_CHUNK"
echo "lora gpu=$GPU_LORA per_phase=$LORA_PER_PHASE ($LORA_STEPS_NOTE)"

echo "[launch] dev40 chunk gate"
(
  set -euo pipefail
  cd /workspace
  CUDA_VISIBLE_DEVICES="$GPU_DEV40" \
  LIMIT="$DEV_LIMIT" \
  CHUNK_FRAMES="$DEV_CHUNK" \
  bash /workspace/wan_ticft/evaluation/run_ltx2b_dev40_chunked_traffic_bdd4_gate.sh
) >"$LOG_DIR/dev40_chunk.log" 2>&1 &
DEV_PID=$!

echo "[launch] camera-style LoRA"
(
  set -euo pipefail
  cd /workspace
  CUDA_VISIBLE_DEVICES="$GPU_LORA" \
  PER_PHASE="$LORA_PER_PHASE" \
  bash /workspace/wan_ticft/evaluation/run_ltx2b_camera_style_lora_bdd100_s300.sh
) >"$LOG_DIR/camera_style_lora.log" 2>&1 &
LORA_PID=$!

echo "dev40_pid=$DEV_PID"
echo "lora_pid=$LORA_PID"
echo
echo "Watch logs:"
echo "  tail -f $LOG_DIR/dev40_chunk.log"
echo "  tail -f $LOG_DIR/camera_style_lora.log"
echo
echo "Wait command:"
echo "  wait $DEV_PID $LORA_PID"

set +e
wait "$DEV_PID"
DEV_STATUS=$?
echo "dev40_exit=$DEV_STATUS"

wait "$LORA_PID"
LORA_STATUS=$?
echo "lora_exit=$LORA_STATUS"
set -e

if [[ "$DEV_STATUS" -ne 0 || "$LORA_STATUS" -ne 0 ]]; then
  echo "One or more jobs failed. See $LOG_DIR." >&2
  exit 1
fi

echo "All jobs completed."
echo "dev40 log: $LOG_DIR/dev40_chunk.log"
echo "lora log: $LOG_DIR/camera_style_lora.log"

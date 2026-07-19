#!/usr/bin/env bash
set -euo pipefail

LOG=/workspace/experiments/wan_ticft/ltxv_bdd_gate_video1170.log
mkdir -p "$(dirname "$LOG")"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-3}" \
bash /workspace/wan_ticft/evaluation/run_ltxv_bdd_gate.sh >"$LOG" 2>&1

echo "Finished. Log saved to: $LOG"
tail -20 "$LOG"

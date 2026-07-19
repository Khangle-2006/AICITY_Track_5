#!/usr/bin/env bash
set -euo pipefail

LOG=/workspace/experiments/wan_ticft/ltxv_bdd_dense6_eval.log
mkdir -p "$(dirname "$LOG")"

bash /workspace/wan_ticft/evaluation/run_ltxv_bdd_dense6_eval.sh >"$LOG" 2>&1

echo "Finished. Log saved to: $LOG"
tail -35 "$LOG"

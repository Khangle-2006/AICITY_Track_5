#!/usr/bin/env bash
set -euo pipefail

LOG=/workspace/experiments/wan_ticft/ltxv_bdd_feasibility.log
mkdir -p "$(dirname "$LOG")"

bash /workspace/wan_ticft/evaluation/run_ltxv_bdd_feasibility.sh >"$LOG" 2>&1

echo "Finished. Log saved to: $LOG"
tail -20 "$LOG"

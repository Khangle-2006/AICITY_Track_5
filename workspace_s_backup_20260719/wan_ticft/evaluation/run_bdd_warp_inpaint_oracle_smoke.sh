#!/usr/bin/env bash
set -euo pipefail

cd /workspace

PYTHON=/opt/conda/envs/wan22/bin/python
ROOT=/workspace/experiments/wan_ticft

echo "=== BDD oracle future-warp + LTX hole-fill smoke ==="
"$PYTHON" -m wan_ticft.evaluation.evaluate_bdd_warp_inpaint_oracle \
  --manifest "$ROOT/ltxv_bdd25_gate/dense40_manifest.jsonl" \
  --ltx_report "$ROOT/ltxv_bdd25_transitions/ltx_transition_16/report.csv" \
  --out_csv "$ROOT/bdd_warp_inpaint_oracle_smoke2.csv" \
  --flow_width 640 \
  --hole_dilate 0 2 4 8 \
  --limit 2

echo
echo "Smoke passes technically only if both samples finish and all candidates are finite."
echo "Run the full gate only after checking runtime, hole fractions, and metric sanity."

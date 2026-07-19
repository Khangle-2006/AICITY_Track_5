#!/usr/bin/env bash
set -euo pipefail

cd /workspace

PYTHON=/opt/conda/envs/wan22/bin/python
ROOT=/workspace/experiments/wan_ticft

echo "=== BDD LTX low-frequency region correction gate ==="
"$PYTHON" -m wan_ticft.evaluation.evaluate_ltx_region_lowfreq \
  --manifest "$ROOT/ltxv_bdd25_gate/dense40_manifest.jsonl" \
  --ltx_report "$ROOT/ltxv_bdd25_transitions/ltx_transition_16/report.csv" \
  --out_csv "$ROOT/ltx_region_lowfreq_gate.csv"

echo
echo "Gate rule:"
echo "  - Continue only if a low-frequency correction improves LTX16 by >=0.15 PSNR,"
echo "    loses no more than 0.002 SSIM, and improves AbsDiff."
echo "  - If pass, materialize only the best variant for visual ghosting/flicker QA."
echo "  - If fail, close frequency-separated compositing without further sigma/alpha tuning."

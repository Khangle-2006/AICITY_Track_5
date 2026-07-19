#!/usr/bin/env bash
set -euo pipefail

cd /workspace

PYTHON=/opt/conda/envs/wan22/bin/python
ROOT=/workspace/experiments/wan_ticft

echo "=== BDD oracle future-warp + LTX hole-fill gate ==="
"$PYTHON" -m wan_ticft.evaluation.evaluate_bdd_warp_inpaint_oracle \
  --manifest "$ROOT/ltxv_bdd25_gate/dense40_manifest.jsonl" \
  --ltx_report "$ROOT/ltxv_bdd25_transitions/ltx_transition_16/report.csv" \
  --out_csv "$ROOT/bdd_warp_inpaint_oracle_gate.csv" \
  --flow_width 640 \
  --hole_dilate 0 2 4 8

echo
echo "Gate rule:"
echo "  - QW2 passes if LTX hole fill beats same-radius repeat fill by >=0.30 PSNR"
echo "    without losing >0.005 SSIM, and the composite beats standalone LTX16."
echo "  - Target-hole fill measures the upper bound of mask-localized inpainting."
echo "  - If target fill beats repeat fill by <0.50 PSNR, holes are not the bottleneck: stop."
echo "  - If target fill has >=1.00 PSNR headroom but LTX fill fails, test one bounded"
echo "    ProPainter/inpainting smoke; do not expand the full layered pipeline yet."
echo "  - Reject useful masks covering >25% of pixels because they no longer isolate disocclusion."

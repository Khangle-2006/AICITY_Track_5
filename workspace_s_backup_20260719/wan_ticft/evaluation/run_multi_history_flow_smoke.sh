#!/usr/bin/env bash
set -euo pipefail

cd /workspace

PYTHON=/opt/conda/envs/wan22/bin/python
ROOT=/workspace/experiments/wan_ticft

echo "=== Multi-history robust-flow smoke: 2 samples/domain ==="
"$PYTHON" -m wan_ticft.evaluation.evaluate_manifest_motion_baselines \
  --manifest "$ROOT/manifests/val_ticft_balanced_125_frames.jsonl" \
  --out_csv "$ROOT/multi_history_flow_smoke6.csv" \
  --candidates region_norm_medium region_multi_history_median \
  --limit_per_domain 2

echo
echo "Gate rule:"
echo "  - Pass only if multi-history median improves macro PSNR by >=0.15,"
echo "    improves macro SSIM, and no domain loses >0.05 PSNR."
echo "  - If fail, allow one correction only: recency-weighted median."
echo "  - If that also fails, close multi-history flow."

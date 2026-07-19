#!/usr/bin/env bash
set -euo pipefail

cd /workspace

PYTHON=/opt/conda/envs/wan22/bin/python
ROOT=/workspace/experiments/wan_ticft

echo "=== Multi-history recency-weighted median: final smoke, 2 samples/domain ==="
"$PYTHON" -m wan_ticft.evaluation.evaluate_manifest_motion_baselines \
  --manifest "$ROOT/manifests/val_ticft_balanced_125_frames.jsonl" \
  --out_csv "$ROOT/multi_history_recency_smoke6.csv" \
  --candidates region_norm_medium region_multi_history_recency \
  --limit_per_domain 2

echo
echo "Final stop rule:"
echo "  - Pass only if recency median improves macro PSNR by >=0.15,"
echo "    improves macro SSIM, and no domain loses >0.05 PSNR."
echo "  - If fail, close multi-history flow. No further weighting/tuning."

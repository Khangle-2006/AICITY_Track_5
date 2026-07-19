#!/usr/bin/env bash
set -euo pipefail

cd /workspace

OUT=/workspace/experiments/wan_ticft/epona_region_keyframe_gate4.csv

echo "=== Offline Epona-keyframe residual into region_norm_medium gate ==="
/opt/conda/envs/wan22/bin/python \
  -m wan_ticft.evaluation.evaluate_epona_region_keyframe_gate \
  --manifest /workspace/experiments/wan_ticft/manifests/val_ticft_balanced_125_frames.jsonl \
  --epona_root /workspace/third_party/Epona/test_videos/gate4 \
  --future_frames 24 \
  --anchor_offset 6 \
  --decay_frames 12 \
  --out_csv "$OUT"

echo
echo "Gate rule:"
echo "  - Pass only if the hybrid improves region_norm_medium by >=0.10 macro PSNR,"
echo "    does not reduce macro SSIM, wins >=3/4, and no sample loses >0.50 PSNR."
echo "  - If fail, allow one correction using a motion-region-masked residual."
echo "  - If correction fails, close all Epona keyframe hybrid work."

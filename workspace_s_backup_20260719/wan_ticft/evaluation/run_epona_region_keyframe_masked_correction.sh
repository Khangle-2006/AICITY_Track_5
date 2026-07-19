#!/usr/bin/env bash
set -euo pipefail

cd /workspace

OUT=/workspace/experiments/wan_ticft/epona_region_keyframe_masked_correction4.csv

echo "=== Final Epona-keyframe correction: motion-region-masked residual ==="
/opt/conda/envs/wan22/bin/python \
  -m wan_ticft.evaluation.evaluate_epona_region_keyframe_gate \
  --manifest /workspace/experiments/wan_ticft/manifests/val_ticft_balanced_125_frames.jsonl \
  --epona_root /workspace/third_party/Epona/test_videos/gate4 \
  --future_frames 24 \
  --anchor_offset 6 \
  --decay_frames 12 \
  --motion_masked \
  --out_csv "$OUT"

echo
echo "Final correction rule:"
echo "  - Pass only if the masked hybrid improves region_norm_medium by >=0.10"
echo "    macro PSNR, does not reduce macro SSIM, wins >=3/4, and no sample"
echo "    loses >0.50 PSNR."
echo "  - If fail, close all Epona keyframe and rollout work."

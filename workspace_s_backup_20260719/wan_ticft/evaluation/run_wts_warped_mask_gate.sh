#!/usr/bin/env bash
set -euo pipefail

cd /workspace

PYTHON=/opt/conda/envs/wan22/bin/python
MANIFEST=/workspace/experiments/wan_ticft/manifests/val_ticft_balanced_125_frames.jsonl

for VIEW in wts_overhead_or_fixed_camera wts_vehicle_or_ip_camera; do
  SHORT=${VIEW#wts_}
  "$PYTHON" -m wan_ticft.evaluation.evaluate_manifest_motion_baselines \
    --manifest "$MANIFEST" \
    --view_type "$VIEW" \
    --source_future_frames 50 \
    --out_csv "/workspace/experiments/wan_ticft/wts_${SHORT}_warped_mask_gate.csv"
done

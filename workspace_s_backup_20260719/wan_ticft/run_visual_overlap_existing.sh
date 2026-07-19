#!/usr/bin/env bash
set -euo pipefail
cd /workspace
export PYTHONPATH=/workspace
/opt/conda/envs/wan22/bin/python /workspace/wan_ticft/scan_test_overlap_visual.py \
  --root /data/Track_5/WTS_dataset/WTS_dataset/external/video \
  --root /data/Track_5/WTS_dataset/WTS_dataset/video \
  --out_csv /workspace/experiments/wan_ticft/test_overlap_visual_existing_wts_external_seek15.csv \
  --stride 15 \
  --topk 8 \
  --refine_radius 20 \
  --seek

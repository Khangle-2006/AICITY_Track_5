#!/usr/bin/env bash
set -euo pipefail
cd /workspace
export PYTHONPATH=/workspace
/opt/conda/envs/wan22/bin/python /workspace/wan_ticft/materialize_fullinput_copy_then_splat.py \
  --manifest /workspace/experiments/wan_ticft/manifests/test_ticft_frames.jsonl \
  --report /workspace/experiments/wan_ticft/hybrid_multiframe_mf_h8_ltxhf_t55_wts_candidate_report.csv \
  --out_dir /workspace/experiments/wan_ticft/hybrid_fullinput_copy_then_splat_d070_candidate \
  --flow_width 640 \
  --decay 0.70 \
  --max_scale 3.0 \
  --hole_dilate 0

#!/usr/bin/env bash
set -euo pipefail

cd /workspace

python -m wan_ticft.evaluation.build_hybrid_submission \
  --index_dir /workspace/experiments/index \
  --wan_dir /workspace/experiments/wan22_submission_ti2v_121f_s10 \
  --region_dir /workspace/experiments/region_norm_medium_submission \
  --out_dir /workspace/experiments/hybrid_wan_region_fixed_submission \
  --report_csv /workspace/experiments/hybrid_wan_region_fixed_verify.csv \
  --zip /workspace/experiments/hybrid_wan_region_fixed_flat.zip \
  --region_views wts_overhead_or_fixed_camera \
  --clean

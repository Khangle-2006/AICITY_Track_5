#!/usr/bin/env bash
set -euo pipefail

cd /workspace

/opt/conda/envs/wan22/bin/python -m wan_ticft.evaluation.build_hybrid_submission \
  --index_dir /workspace/experiments/index \
  --wan_dir /workspace/experiments/wan_ticft/ltxv_bdd_test_full \
  --wan_label ltx_video_2b \
  --region_dir /workspace/experiments/region_norm_medium_submission \
  --region_label region_norm_medium \
  --out_dir /workspace/experiments/hybrid_ltx_region_wts_submission \
  --report_csv /workspace/experiments/hybrid_ltx_region_wts_verify.csv \
  --region_views wts_overhead_or_fixed_camera wts_vehicle_or_ip_camera \
  --clean

/opt/conda/envs/wan22/bin/python -m baseline.submission \
  --index_dir /workspace/experiments/index \
  --gen_dir /workspace/experiments/hybrid_ltx_region_wts_submission \
  --submission_dir /workspace/experiments/hybrid_ltx_region_wts_submission_verified \
  --report_csv /workspace/experiments/hybrid_ltx_region_wts_submission_verify.csv \
  --verify_only

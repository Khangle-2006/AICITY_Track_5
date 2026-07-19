#!/usr/bin/env bash
set -euo pipefail

cd /workspace

BDD_OUT=/workspace/experiments/wan_ticft/ltxv_bdd_test_transition16
SUBMISSION=/workspace/experiments/hybrid_ltx_transition16_region_wts_submission

/opt/conda/envs/wan22/bin/python -m wan_ticft.evaluation.build_ltx_test_transition \
  --test_csv /workspace/experiments/index/test_index.csv \
  --ltx_dir /workspace/experiments/wan_ticft/ltxv_bdd_test_full \
  --out_dir "$BDD_OUT" \
  --transition_frames 16 \
  --clean

/opt/conda/envs/wan22/bin/python -m baseline.video_contact_sheet \
  --input "$BDD_OUT" \
  --mode frames \
  --out_dir "${BDD_OUT}_sheets" \
  --samples 16 \
  --tile_width 320 \
  --cols 4

/opt/conda/envs/wan22/bin/python -m wan_ticft.evaluation.build_hybrid_submission \
  --index_dir /workspace/experiments/index \
  --wan_dir "$BDD_OUT" \
  --wan_label ltx_transition16 \
  --region_dir /workspace/experiments/region_norm_medium_submission \
  --region_label region_norm_medium \
  --out_dir "$SUBMISSION" \
  --report_csv /workspace/experiments/hybrid_ltx_transition16_region_wts_verify.csv \
  --region_views wts_overhead_or_fixed_camera wts_vehicle_or_ip_camera \
  --clean

/opt/conda/envs/wan22/bin/python -m baseline.submission \
  --index_dir /workspace/experiments/index \
  --gen_dir "$SUBMISSION" \
  --submission_dir /workspace/experiments/hybrid_ltx_transition16_region_wts_submission_verified \
  --report_csv /workspace/experiments/hybrid_ltx_transition16_region_wts_submission_verify.csv \
  --verify_only

#!/usr/bin/env bash
set -euo pipefail

cd /workspace

PYTHON=/opt/conda/envs/wan22/bin/python
INDEX=/workspace/experiments/index
BDD=/workspace/experiments/wan_ticft/ltxv_bdd_test_transition16
WTS=/workspace/experiments/wan_ticft/wts_overhead_long24_vehicle_medium_test
SUBMISSION=/workspace/experiments/hybrid_ltx_transition16_wts_domain_routed_candidate

# Generate only the validated domain routes. The output directory is unique
# so no stale routes from older candidates can enter this build.
"$PYTHON" -m baseline.cli \
  --index_dir "$INDEX" \
  --out "$WTS" \
  --view_type wts_overhead_or_fixed_camera \
  --method region_norm_medium \
  --region_norm_decay 0.995 \
  --region_norm_max_flow_scale 24.0

"$PYTHON" -m baseline.cli \
  --index_dir "$INDEX" \
  --out "$WTS" \
  --view_type wts_vehicle_or_ip_camera \
  --method region_norm_medium \
  --region_norm_decay 0.96 \
  --region_norm_max_flow_scale 6.0

"$PYTHON" -m wan_ticft.evaluation.build_hybrid_submission \
  --index_dir "$INDEX" \
  --wan_dir "$BDD" \
  --wan_label ltx_transition16 \
  --region_dir "$WTS" \
  --region_label wts_overhead_long24_vehicle_medium \
  --out_dir "$SUBMISSION" \
  --report_csv /workspace/experiments/hybrid_ltx_transition16_wts_domain_routed_candidate_verify.csv \
  --region_views wts_overhead_or_fixed_camera wts_vehicle_or_ip_camera \
  --clean

"$PYTHON" -m baseline.submission \
  --index_dir "$INDEX" \
  --gen_dir "$SUBMISSION" \
  --submission_dir /workspace/experiments/hybrid_ltx_transition16_wts_domain_routed_candidate_verified \
  --report_csv /workspace/experiments/hybrid_ltx_transition16_wts_domain_routed_candidate_submission_verify.csv \
  --verify_only

echo
echo "Candidate verified but not zipped:"
echo "  $SUBMISSION"

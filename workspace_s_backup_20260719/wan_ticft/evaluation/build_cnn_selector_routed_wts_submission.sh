#!/usr/bin/env bash
set -euo pipefail

cd /workspace

PYTHON=/opt/conda/envs/wan22/bin/python
INDEX=/workspace/experiments/index
BDD=/workspace/experiments/wan_ticft/cnn_ltx_region_selector_bdd_test
WTS=/workspace/experiments/wan_ticft/region_norm_routed_wts_test
SUBMISSION=/workspace/experiments/hybrid_cnn_selector_routed_wts_submission

# Fixed cameras benefit from sustained motion over the long test horizon.
"$PYTHON" -m baseline.cli \
  --index_dir "$INDEX" \
  --out "$WTS" \
  --view_type wts_overhead_or_fixed_camera \
  --method region_norm_medium \
  --region_norm_decay 0.995 \
  --region_norm_max_flow_scale 24.0

# Vehicle/IP cameras need a more conservative long-horizon route.
"$PYTHON" -m baseline.cli \
  --index_dir "$INDEX" \
  --out "$WTS" \
  --view_type wts_vehicle_or_ip_camera \
  --method region_norm_medium \
  --region_norm_decay 0.985 \
  --region_norm_max_flow_scale 12.0

# Validation supports phase-aware routing for vehicle/IP cameras.
"$PYTHON" -m baseline.cli \
  --index_dir "$INDEX" \
  --out "$WTS" \
  --view_type wts_vehicle_or_ip_camera \
  --phase 0 \
  --method region_norm_medium \
  --region_norm_decay 0.96 \
  --region_norm_max_flow_scale 6.0

for PHASE in 3 4; do
  "$PYTHON" -m baseline.cli \
    --index_dir "$INDEX" \
    --out "$WTS" \
    --view_type wts_vehicle_or_ip_camera \
    --phase "$PHASE" \
    --method region_norm_medium \
    --region_norm_decay 0.995 \
    --region_norm_max_flow_scale 24.0
done

"$PYTHON" -m baseline.video_contact_sheet \
  --input "$WTS" \
  --mode frames \
  --out_dir "${WTS}_sheets" \
  --samples 16 \
  --tile_width 320 \
  --cols 4

"$PYTHON" -m wan_ticft.evaluation.build_hybrid_submission \
  --index_dir "$INDEX" \
  --wan_dir "$BDD" \
  --wan_label cnn_ltx_region_selector \
  --region_dir "$WTS" \
  --region_label region_norm_routed_long_horizon \
  --out_dir "$SUBMISSION" \
  --report_csv /workspace/experiments/hybrid_cnn_selector_routed_wts_verify.csv \
  --region_views wts_overhead_or_fixed_camera wts_vehicle_or_ip_camera \
  --clean

"$PYTHON" -m baseline.submission \
  --index_dir "$INDEX" \
  --gen_dir "$SUBMISSION" \
  --submission_dir /workspace/experiments/hybrid_cnn_selector_routed_wts_submission_verified \
  --report_csv /workspace/experiments/hybrid_cnn_selector_routed_wts_submission_verify.csv \
  --verify_only

echo "Inspect WTS before packaging: ${WTS}_sheets"

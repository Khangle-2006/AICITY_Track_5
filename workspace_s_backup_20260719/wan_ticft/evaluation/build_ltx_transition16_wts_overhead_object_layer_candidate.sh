#!/usr/bin/env bash
set -euo pipefail

cd /workspace

PYTHON=${PYTHON:-/opt/conda/envs/wan22/bin/python}
INDEX=${INDEX:-/workspace/experiments/index}
BDD=${BDD:-/workspace/experiments/wan_ticft/ltxv_bdd_test_transition16}
WTS=${WTS:-/workspace/experiments/wan_ticft/wts_overhead_object_layer_vehicle_medium_test}
SUBMISSION=${SUBMISSION:-/workspace/experiments/hybrid_ltx_transition16_wts_overhead_object_layer_candidate}
REPORT=${REPORT:-/workspace/experiments/wan_ticft/hybrid_ltx_transition16_wts_overhead_object_layer_candidate_report.csv}

echo "=== Build BDD transition16 + WTS overhead object-layer + vehicle medium candidate ==="
echo "BDD=${BDD}"
echo "WTS=${WTS}"
echo "SUBMISSION=${SUBMISSION}"

if [[ ! -d "$BDD" ]]; then
  echo "Missing BDD transition16 directory: $BDD" >&2
  exit 2
fi

rm -rf "$WTS" "$SUBMISSION"

echo
echo "[1/4] Generate WTS overhead/fixed with small object-layer fill"
"$PYTHON" -m baseline.cli \
  --index_dir "$INDEX" \
  --out "$WTS" \
  --view_type wts_overhead_or_fixed_camera \
  --method actor_layer_small_filled \
  --actor_layer_decay 0.985 \
  --actor_layer_max_flow_scale 12.0 \
  --actor_layer_max_area_frac 0.025

echo
echo "[2/4] Generate WTS vehicle/IP with current medium region flow"
"$PYTHON" -m baseline.cli \
  --index_dir "$INDEX" \
  --out "$WTS" \
  --view_type wts_vehicle_or_ip_camera \
  --method region_norm_medium \
  --region_norm_decay 0.96 \
  --region_norm_max_flow_scale 6.0

echo
echo "[3/4] Build hybrid submission directory, no zip"
"$PYTHON" -m wan_ticft.evaluation.build_hybrid_submission \
  --index_dir "$INDEX" \
  --wan_dir "$BDD" \
  --region_dir "$WTS" \
  --out_dir "$SUBMISSION" \
  --report_csv "$REPORT" \
  --wan_label ltx_transition16 \
  --region_label wts_overhead_object_layer_vehicle_medium \
  --region_views wts_overhead_or_fixed_camera wts_vehicle_or_ip_camera \
  --clean

echo
echo "[4/4] Verify full submission folder"
"$PYTHON" -m baseline.submission \
  --index_dir "$INDEX" \
  --gen_dir "$SUBMISSION" \
  --submission_dir "$SUBMISSION" \
  --verify_only

echo
echo "Candidate verified but not zipped:"
echo "  $SUBMISSION"
echo "Report:"
echo "  $REPORT"

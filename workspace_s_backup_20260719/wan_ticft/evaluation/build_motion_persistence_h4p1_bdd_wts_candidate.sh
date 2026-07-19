#!/usr/bin/env bash
set -euo pipefail

cd /workspace

PYTHON=/opt/conda/envs/wan22/bin/python
INDEX=/workspace/experiments/index
TEST_CSV="$INDEX/test_index.csv"
ROOT=/workspace/experiments/wan_ticft
BDD_DEPTH="$ROOT/depth_layered_a75_bdd_test"
BDD_ROUTED="$ROOT/bdd_motion_persistence_h4p1_bdd_test"
WTS="$ROOT/wts_overhead_object_layer_vehicle_medium_test"
SUBMISSION=/workspace/experiments/hybrid_bdd_motion_persistence_h4p1_wts_overhead_object_candidate
REPORT="$ROOT/hybrid_bdd_motion_persistence_h4p1_wts_overhead_object_candidate_report.csv"

echo "=== Build BDD motion-persistence h4/p1 + current WTS candidate ==="
echo "CPU-only router; no zip is created."
for path in "$TEST_CSV" "$BDD_DEPTH/report.csv"; do
  [[ -s "$path" ]] || { echo "Missing input: $path" >&2; exit 2; }
done
[[ -d "$WTS" ]] || { echo "Missing current WTS source: $WTS" >&2; exit 2; }

echo "[1/3] Route BDD test: repeat only for low-motion h<=4 p<=1"
"$PYTHON" -m wan_ticft.evaluation.materialize_motion_persistence_router_test \
  --test_csv "$TEST_CSV" \
  --depth_dir "$BDD_DEPTH" \
  --out_dir "$BDD_ROUTED" \
  --view_type bdd \
  --hist_threshold 4.0 \
  --pred_threshold 1.0 \
  --motion_frames 12 \
  --clean

echo "[2/3] Build hybrid with unchanged WTS/object-layer outputs"
"$PYTHON" -m wan_ticft.evaluation.build_hybrid_submission \
  --index_dir "$INDEX" \
  --wan_dir "$BDD_ROUTED" \
  --region_dir "$WTS" \
  --out_dir "$SUBMISSION" \
  --report_csv "$REPORT" \
  --wan_label bdd_motion_persistence_h4p1 \
  --region_label wts_overhead_object_layer_vehicle_medium \
  --region_views wts_overhead_or_fixed_camera wts_vehicle_or_ip_camera \
  --clean

echo "[3/3] Verify complete 71-sample folder"
"$PYTHON" -m baseline.submission \
  --index_dir "$INDEX" \
  --gen_dir "$SUBMISSION" \
  --submission_dir "$SUBMISSION" \
  --verify_only

echo
echo "Candidate verified but not zipped:"
echo "  $SUBMISSION"
echo "Hybrid report:"
echo "  $REPORT"
echo "BDD route diagnostics:"
echo "  $BDD_ROUTED/diagnostics.csv"

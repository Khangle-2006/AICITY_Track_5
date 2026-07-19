#!/usr/bin/env bash
set -euo pipefail

cd /workspace

PYTHON=/opt/conda/envs/wan22/bin/python
INDEX=/workspace/experiments/index
TEST_CSV="$INDEX/test_index.csv"
ROOT=/workspace/experiments/wan_ticft
BDD_BASE="$ROOT/cnn_ltxmix65_bdd_test"
BDD_A75="$ROOT/depth_layered_a75_bdd_test"
BDD_A50="$ROOT/depth_layered_a50_bdd_test"
WTS="$ROOT/wts_overhead_object_layer_vehicle_medium_test"
SUBMISSION=/workspace/experiments/hybrid_depth_layered_a50_wts_overhead_object_layer_candidate
REPORT="$ROOT/hybrid_depth_layered_a50_wts_overhead_object_layer_candidate_report.csv"

echo "=== Build last-minute BDD depth-a50 candidate from existing lossless outputs ==="
echo "No inference and no zip are performed."
for path in "$TEST_CSV" "$BDD_BASE" "$BDD_A75" "$WTS"; do
  [[ -e "$path" ]] || { echo "Missing input: $path" >&2; exit 2; }
done

echo "[1/3] Interpolate a75 residual to a50 (ratio 2/3)"
"$PYTHON" -m wan_ticft.evaluation.interpolate_test_candidate \
  --test_csv "$TEST_CSV" \
  --base_dir "$BDD_BASE" \
  --candidate_dir "$BDD_A75" \
  --out_dir "$BDD_A50" \
  --view_type bdd \
  --ratio 0.6666666667 \
  --clean

echo "[2/3] Build BDD-a50 plus unchanged WTS folder"
"$PYTHON" -m wan_ticft.evaluation.build_hybrid_submission \
  --index_dir "$INDEX" \
  --wan_dir "$BDD_A50" \
  --region_dir "$WTS" \
  --out_dir "$SUBMISSION" \
  --report_csv "$REPORT" \
  --wan_label depth_layered_a50 \
  --region_label wts_overhead_object_layer_vehicle_medium \
  --region_views wts_overhead_or_fixed_camera wts_vehicle_or_ip_camera \
  --clean

echo "[3/3] Verify complete submission folder"
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

#!/bin/bash
set -euo pipefail
cd /workspace
PYTHON=${PYTHON:-/opt/conda/envs/wan22/bin/python}
INDEX=/workspace/experiments/index
ROOT=/workspace/experiments/wan_ticft
BDD=$ROOT/ltx2b_chunked_1280x720_bdd_test
WTS=$ROOT/wts_overhead_object_layer_vehicle_medium_test
SUBMISSION=/workspace/experiments/hybrid_ltx2b_chunked_1280x720_wts_candidate
REPORT=$ROOT/hybrid_ltx2b_chunked_1280x720_wts_candidate_report.csv

echo "=== Build LTX2B chunked 1280x720 (BDD) + WTS object-layer submission ==="
echo "BDD=$BDD"; echo "WTS=$WTS"; echo "SUBMISSION=$SUBMISSION"
for p in "$INDEX/test_index.csv" "$BDD/report.csv" "$WTS"; do
  [[ -e "$p" ]] || { echo "Missing input: $p" >&2; exit 2; }
done

echo "[1/2] Merge BDD + WTS into 71-folder submission"
"$PYTHON" -m wan_ticft.evaluation.build_hybrid_submission \
  --index_dir "$INDEX" \
  --wan_dir "$BDD" \
  --region_dir "$WTS" \
  --out_dir "$SUBMISSION" \
  --report_csv "$REPORT" \
  --wan_label ltx2b_chunked_1280x720 \
  --region_label wts_overhead_object_layer_vehicle_medium \
  --region_views wts_overhead_or_fixed_camera wts_vehicle_or_ip_camera \
  --clean

echo "[2/2] Verify complete submission (71 samples, frame counts)"
"$PYTHON" -m baseline.submission \
  --index_dir "$INDEX" \
  --gen_dir "$SUBMISSION" \
  --submission_dir "$SUBMISSION" \
  --verify_only

echo
echo "Submission ready (not zipped): $SUBMISSION"
echo "Report: $REPORT"

#!/usr/bin/env bash
set -euo pipefail

cd /workspace

PYTHON=${PYTHON:-/opt/conda/envs/wan22/bin/python}
INDEX=${INDEX:-/workspace/experiments/index}
ROOT=${ROOT:-/workspace/experiments/wan_ticft}
BDD=${BDD:-$ROOT/ltx2b_chunked_traffic_bdd_test}
WTS=${WTS:-$ROOT/wts_overhead_object_layer_vehicle_medium_test}
SUBMISSION=${SUBMISSION:-/workspace/experiments/hybrid_ltx2b_chunked_bdd_wts_object_vehicle_candidate}
REPORT=${REPORT:-$ROOT/hybrid_ltx2b_chunked_bdd_wts_object_vehicle_candidate_report.csv}
ZIP=${ZIP:-}

echo "=== Build LTX2B chunked BDD + current WTS candidate ==="
echo "BDD=$BDD"
echo "WTS=$WTS"
echo "SUBMISSION=$SUBMISSION"

for path in "$INDEX/test_index.csv" "$BDD/report.csv" "$WTS"; do
  [[ -e "$path" ]] || { echo "Missing input: $path" >&2; exit 2; }
done

COMMON_ARGS=(
  -m wan_ticft.evaluation.build_hybrid_submission
  --index_dir "$INDEX"
  --wan_dir "$BDD"
  --region_dir "$WTS"
  --out_dir "$SUBMISSION"
  --report_csv "$REPORT"
  --wan_label ltx2b_chunked_traffic_bdd
  --region_label wts_overhead_object_layer_vehicle_medium
  --region_views wts_overhead_or_fixed_camera wts_vehicle_or_ip_camera
  --clean
)
if [[ -n "$ZIP" ]]; then
  COMMON_ARGS+=(--zip "$ZIP")
fi
"$PYTHON" "${COMMON_ARGS[@]}"

"$PYTHON" -m baseline.submission \
  --index_dir "$INDEX" \
  --gen_dir "$SUBMISSION" \
  --submission_dir "$SUBMISSION" \
  --verify_only

echo
echo "Candidate verified:"
echo "  $SUBMISSION"
echo "Report:"
echo "  $REPORT"
if [[ -n "$ZIP" ]]; then
  echo "Zip:"
  echo "  $ZIP"
fi

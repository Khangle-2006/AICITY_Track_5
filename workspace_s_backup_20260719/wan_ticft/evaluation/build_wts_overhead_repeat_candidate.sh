#!/bin/bash
set -euo pipefail
# WTS routing: overhead/fixed -> repeat_last (last observed frame; +3 PSNR local), vehicle/ip -> object-layer.
# BDD half = a75 (isolates WTS lever vs 69.24 baseline). Override BDD= to combine with chunk-base later.
cd /workspace
PYTHON=${PYTHON:-/opt/conda/envs/wan22/bin/python}
INDEX=/workspace/experiments/index
TEST_CSV="$INDEX/test_index.csv"
ROOT=/workspace/experiments/wan_ticft
BDD=${BDD:-$ROOT/depth_layered_a75_bdd_test}
WTS=${WTS:-$ROOT/wts_overhead_object_layer_vehicle_medium_test}
REPEAT_OVH=${REPEAT_OVH:-$ROOT/repeat_last_wts_overhead_test}
TAG=${TAG:-a75bdd}
SUBMISSION=/workspace/experiments/hybrid_${TAG}_wts_overhead_repeat_candidate
REPORT=$ROOT/hybrid_${TAG}_wts_overhead_repeat_candidate_report.csv

echo "=== WTS overhead->repeat candidate (BDD=$(basename $BDD)) ==="
for p in "$TEST_CSV" "$BDD/report.csv" "$WTS"; do [[ -e "$p" ]] || { echo "Missing: $p" >&2; exit 2; }; done

echo "[1/3] repeat_last (last observed frame) for WTS overhead/fixed test"
"$PYTHON" -m wan_ticft.evaluation.materialize_repeat_last_test \
  --test_csv "$TEST_CSV" --reference_dir "$WTS" --out_dir "$REPEAT_OVH" \
  --view_type wts_overhead_or_fixed_camera --clean

echo "[2/3] Merge: BDD=wan, vehicle=object-layer, overhead=repeat(override)"
"$PYTHON" -m wan_ticft.evaluation.build_hybrid_submission \
  --index_dir "$INDEX" --wan_dir "$BDD" --region_dir "$WTS" \
  --out_dir "$SUBMISSION" --report_csv "$REPORT" \
  --wan_label "${TAG}_bdd" --region_label wts_vehicle_object_layer \
  --region_views wts_vehicle_or_ip_camera \
  --view_override "wts_overhead_or_fixed_camera=$REPEAT_OVH=repeat_overhead" --clean

echo "[3/3] Verify"
"$PYTHON" -m baseline.submission --index_dir "$INDEX" --gen_dir "$SUBMISSION" --submission_dir "$SUBMISSION" --verify_only
echo; echo "Submission ready: $SUBMISSION"; echo "Report: $REPORT"

#!/usr/bin/env bash
set -euo pipefail

cd /workspace

MODE="${1:-strict}"
PYTHON=/opt/conda/envs/wan22/bin/python
INDEX=/workspace/experiments/index
ROOT=/workspace/experiments/wan_ticft
BDD_DEPTH="$ROOT/depth_layered_a75_bdd_test"
BDD_A125="$ROOT/sa2va_actor_depth_a125_bdd_test"
BDD_DIAG="$BDD_DEPTH/diagnostics.csv"
WTS="$ROOT/wts_overhead_object_layer_vehicle_medium_test"

BDD_ROUTED="$ROOT/selective_sa2va_actor_depth_a125_${MODE}_bdd_test"
ROUTE_CSV="$ROOT/selective_sa2va_actor_depth_a125_${MODE}_route.csv"
SUBMISSION="/workspace/experiments/hybrid_selective_sa2va_actor_depth_a125_${MODE}_wts_overhead_object_candidate"
REPORT="$ROOT/hybrid_selective_sa2va_actor_depth_a125_${MODE}_wts_overhead_object_candidate_report.csv"

echo "=== Selective Sa2VA actor-depth a125 candidate (${MODE}) ==="
echo "CPU-only copy router; no GPU inference and no zip."

for path in \
  "$BDD_DEPTH/report.csv" \
  "$BDD_A125/report.csv" \
  "$BDD_DIAG"; do
  [[ -s "$path" ]] || { echo "Missing input: $path" >&2; exit 2; }
done
[[ -d "$WTS" ]] || { echo "Missing WTS source directory: $WTS" >&2; exit 2; }

case "$MODE" in
  strict|balanced|fidlean) ;;
  *)
    echo "Invalid mode: $MODE (expected strict, balanced, or fidlean)" >&2
    exit 2
    ;;
esac

echo "[1/3] Route BDD between depth_a75 and actor_depth_a125"
"$PYTHON" -m wan_ticft.evaluation.materialize_selective_bdd_route \
  --depth_dir "$BDD_DEPTH" \
  --actor_dir "$BDD_A125" \
  --diagnostics_csv "$BDD_DIAG" \
  --out_dir "$BDD_ROUTED" \
  --route_csv "$ROUTE_CSV" \
  --mode "$MODE" \
  --clean

echo "[2/3] Build hybrid with unchanged WTS"
"$PYTHON" -m wan_ticft.evaluation.build_hybrid_submission \
  --index_dir "$INDEX" \
  --wan_dir "$BDD_ROUTED" \
  --region_dir "$WTS" \
  --out_dir "$SUBMISSION" \
  --report_csv "$REPORT" \
  --wan_label "selective_${MODE}_a125_depth_a75" \
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
echo "BDD route report:"
echo "  $ROUTE_CSV"

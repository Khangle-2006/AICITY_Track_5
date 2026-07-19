#!/usr/bin/env bash
set -euo pipefail

cd /workspace

PYTHON=/opt/conda/envs/wan22/bin/python
INDEX=/workspace/experiments/index
ROOT=/workspace/experiments/wan_ticft
BDD_BASE="$ROOT/cnn_ltxmix65_bdd_test"
BDD_A125="$ROOT/sa2va_actor_depth_a125_bdd_test"
BDD_A150="$ROOT/sa2va_actor_depth_a150_bdd_test"
WTS="$ROOT/wts_overhead_object_layer_vehicle_medium_test"
SUBMISSION=/workspace/experiments/hybrid_sa2va_actor_depth_a150_wts_overhead_object_candidate
REPORT="$ROOT/hybrid_sa2va_actor_depth_a150_wts_overhead_object_candidate_report.csv"

echo "=== Build actor-depth a150 candidate ==="
echo "CPU-only residual rescale from existing a125; no GPU inference and no zip."

for path in "$BDD_BASE/report.csv" "$BDD_A125/report.csv"; do
  [[ -s "$path" ]] || { echo "Missing input: $path" >&2; exit 2; }
done
[[ -d "$WTS" ]] || { echo "Missing WTS source directory: $WTS" >&2; exit 2; }

echo "[1/3] Rescale actor-aware residual from a125 -> a150"
"$PYTHON" -m wan_ticft.evaluation.materialize_actor_depth_rescale_test \
  --test_csv "$INDEX/test_index.csv" \
  --base_dir "$BDD_BASE" \
  --source_dir "$BDD_A125" \
  --out_dir "$BDD_A150" \
  --ratio 1.2 \
  --clean

echo "[2/3] Build hybrid with unchanged WTS"
"$PYTHON" -m wan_ticft.evaluation.build_hybrid_submission \
  --index_dir "$INDEX" \
  --wan_dir "$BDD_A150" \
  --region_dir "$WTS" \
  --out_dir "$SUBMISSION" \
  --report_csv "$REPORT" \
  --wan_label sa2va_actor_depth_a150 \
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
echo "Report:"
echo "  $REPORT"

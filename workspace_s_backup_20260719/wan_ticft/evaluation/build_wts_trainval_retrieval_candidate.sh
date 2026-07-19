#!/usr/bin/env bash
set -euo pipefail

cd /workspace

MODE="${1:-strict}"
PYTHON=/opt/conda/envs/wan22/bin/python
INDEX=/workspace/experiments/index
ROOT=/workspace/experiments/wan_ticft
MATCHES="$ROOT/sequence_retrieval_audit/matches.csv"
WTS_BASE="$ROOT/wts_overhead_object_layer_vehicle_medium_test"
BDD_DIR="${BDD_DIR:-$ROOT/depth_layered_a75_bdd_test}"
BDD_LABEL="${BDD_LABEL:-depth_layered_a75}"
WTS_OUT="$ROOT/wts_trainval_retrieval_overhead_${MODE}_test"
ROUTE_CSV="$ROOT/wts_trainval_retrieval_overhead_${MODE}_route.csv"
SHEETS="$ROOT/wts_trainval_retrieval_overhead_${MODE}_sheets"
SUBMISSION="/workspace/experiments/hybrid_${BDD_LABEL}_wts_trainval_retrieval_${MODE}_candidate"
REPORT="$ROOT/hybrid_${BDD_LABEL}_wts_trainval_retrieval_${MODE}_candidate_report.csv"

echo "=== WTS train/val retrieval candidate (${MODE}) ==="
echo "Uses only official train/val reference videos; no test future extraction and no zip."
echo "BDD_DIR=$BDD_DIR"

case "$MODE" in
  ultra|strict|loose) ;;
  *)
    echo "Invalid mode: $MODE (expected ultra, strict, or loose)" >&2
    exit 2
    ;;
esac

for path in "$MATCHES" "$BDD_DIR/report.csv"; do
  [[ -s "$path" ]] || { echo "Missing input: $path" >&2; exit 2; }
done
[[ -d "$WTS_BASE" ]] || { echo "Missing WTS fallback directory: $WTS_BASE" >&2; exit 2; }

echo "[1/3] Materialize WTS retrieval/fallback folder"
"$PYTHON" -m wan_ticft.evaluation.materialize_wts_trainval_retrieval_test \
  --index_dir "$INDEX" \
  --matches_csv "$MATCHES" \
  --fallback_dir "$WTS_BASE" \
  --out_dir "$WTS_OUT" \
  --route_csv "$ROUTE_CSV" \
  --sheets_dir "$SHEETS" \
  --mode "$MODE" \
  --clean

echo "[2/3] Build hybrid with BDD + WTS retrieval"
"$PYTHON" -m wan_ticft.evaluation.build_hybrid_submission \
  --index_dir "$INDEX" \
  --wan_dir "$BDD_DIR" \
  --region_dir "$WTS_OUT" \
  --out_dir "$SUBMISSION" \
  --report_csv "$REPORT" \
  --wan_label "$BDD_LABEL" \
  --region_label "wts_trainval_retrieval_${MODE}" \
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
echo "WTS route report:"
echo "  $ROUTE_CSV"
echo "Sheets:"
echo "  $SHEETS"

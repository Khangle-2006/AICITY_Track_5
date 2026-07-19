#!/usr/bin/env bash
set -euo pipefail

cd /workspace

PYTHON=${PYTHON:-/opt/conda/envs/wan22/bin/python}
INDEX=${INDEX:-/workspace/experiments/index}
TEST_CSV=${TEST_CSV:-$INDEX/test_index.csv}
ROOT=${ROOT:-/workspace/experiments/wan_ticft}

BDD_DEPTH=${BDD_DEPTH:-$ROOT/depth_layered_a75_bdd_test}
BDD_REPEAT=${BDD_REPEAT:-$ROOT/repeat_last_bdd_test_for_phase12}
BDD_REGION=${BDD_REGION:-$ROOT/region_norm_medium_bdd_test_for_guarded_phase}
BDD_GUARDED=${BDD_GUARDED:-$ROOT/bdd_phase0_depth_phase12_repeat_phase34_region_test}
WTS=${WTS:-$ROOT/wts_overhead_object_layer_vehicle_medium_test}

SUBMISSION=${SUBMISSION:-/workspace/experiments/hybrid_bdd_guarded_phase_region_wts_object_vehicle_candidate}
REPORT=${REPORT:-$ROOT/hybrid_bdd_guarded_phase_region_wts_object_vehicle_candidate_report.csv}
ZIP=${ZIP:-}

echo "=== Build BDD guarded phase-region + current WTS candidate ==="
echo "Rule:"
echo "  BDD phase 0   -> depth_a75"
echo "  BDD phase 1/2 -> repeat_last"
echo "  BDD phase 3/4 -> region_norm_medium"
echo "BDD_DEPTH=$BDD_DEPTH"
echo "BDD_REPEAT=$BDD_REPEAT"
echo "BDD_REGION=$BDD_REGION"
echo "BDD_GUARDED=$BDD_GUARDED"
echo "WTS=$WTS"
echo "SUBMISSION=$SUBMISSION"
if [[ -n "$ZIP" ]]; then
  echo "ZIP=$ZIP"
else
  echo "No zip is created."
fi

for path in "$TEST_CSV" "$BDD_DEPTH/report.csv" "$WTS"; do
  [[ -e "$path" ]] || { echo "Missing input: $path" >&2; exit 2; }
done

echo
echo "[1/5] Materialize/reuse BDD repeat-last"
if [[ -s "$BDD_REPEAT/report.csv" ]]; then
  echo "  found existing $BDD_REPEAT/report.csv; skip"
else
  "$PYTHON" -m wan_ticft.evaluation.materialize_repeat_last_test \
    --test_csv "$TEST_CSV" \
    --reference_dir "$BDD_DEPTH" \
    --out_dir "$BDD_REPEAT" \
    --view_type bdd \
    --clean
fi

echo
echo "[2/5] Materialize/reuse BDD region_norm_medium"
if [[ -s "$BDD_REGION/report.csv" ]]; then
  echo "  found existing $BDD_REGION/report.csv; skip"
else
  "$PYTHON" -m baseline.cli \
    --index_dir "$INDEX" \
    --out "$BDD_REGION" \
    --view_type bdd \
    --method region_norm_medium \
    --region_norm_decay 0.96 \
    --region_norm_max_flow_scale 6.0
fi

echo
echo "[3/5] Route BDD by guarded phase rule"
"$PYTHON" -m wan_ticft.evaluation.materialize_phase_rule_test \
  --test_csv "$TEST_CSV" \
  --out_dir "$BDD_GUARDED" \
  --default_dir "$BDD_REGION" \
  --default_label region_norm_medium \
  --phase_dir "0=$BDD_DEPTH=depth_a75" \
  --phase_dir "1=$BDD_REPEAT=repeat_last" \
  --phase_dir "2=$BDD_REPEAT=repeat_last" \
  --view_type bdd \
  --clean

echo
echo "[4/5] Build full hybrid submission folder"
COMMON_ARGS=(
  -m wan_ticft.evaluation.build_hybrid_submission
  --index_dir "$INDEX"
  --wan_dir "$BDD_GUARDED"
  --region_dir "$WTS"
  --out_dir "$SUBMISSION"
  --report_csv "$REPORT"
  --wan_label bdd_phase0_depth_phase12_repeat_phase34_region
  --region_label wts_overhead_object_layer_vehicle_medium
  --region_views wts_overhead_or_fixed_camera wts_vehicle_or_ip_camera
  --clean
)
if [[ -n "$ZIP" ]]; then
  COMMON_ARGS+=(--zip "$ZIP")
fi
"$PYTHON" "${COMMON_ARGS[@]}"

echo
echo "[5/5] Verify complete submission folder"
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
echo "BDD guarded report:"
echo "  $BDD_GUARDED/report.csv"
if [[ -n "$ZIP" ]]; then
  echo "Zip:"
  echo "  $ZIP"
fi

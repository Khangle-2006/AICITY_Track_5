#!/usr/bin/env bash
set -euo pipefail

cd /workspace

PYTHON=${PYTHON:-/opt/conda/envs/wan22/bin/python}
INDEX=${INDEX:-/workspace/experiments/index}
TEST_CSV=${TEST_CSV:-$INDEX/test_index.csv}
ROOT=${ROOT:-/workspace/experiments/wan_ticft}
MIX=${MIX:-0.60}
MIX_LABEL=${MIX_LABEL:-$(python - <<PY
mix=float("$MIX")
print(f"mix{int(round(mix*100)):02d}")
PY
)}
BDD_MIX=${BDD_MIX:-$ROOT/cnn_ltx${MIX_LABEL}_bdd_test}
CNN=${CNN:-$ROOT/cnn_ltx_region_selector_bdd_test}
LTX=${LTX:-$ROOT/ltxv_bdd_test_transition16}
WTS_GUARD=${WTS_GUARD:-$ROOT/wts_guard015_test}
SUBMISSION=${SUBMISSION:-/workspace/experiments/hybrid_cnn_ltx${MIX_LABEL}_wts_guard015_candidate}
REPORT=${REPORT:-$ROOT/hybrid_cnn_ltx${MIX_LABEL}_wts_guard015_candidate_report.csv}

echo "=== Build CNN ${MIX_LABEL} + WTS guard015 candidate ==="

"$PYTHON" -m wan_ticft.evaluation.materialize_cnn_ltxmix_test \
  --test_csv "$TEST_CSV" \
  --cnn_dir "$CNN" \
  --ltx_dir "$LTX" \
  --out_dir "$BDD_MIX" \
  --ltx_mix "$MIX" \
  --clean

if [[ ! -d "$WTS_GUARD" ]]; then
  echo "Missing WTS guard directory: $WTS_GUARD" >&2
  echo "Run build_cnn_ltxmix65_wts_guard015_candidate.sh once, or generate guard WTS first." >&2
  exit 2
fi

"$PYTHON" -m wan_ticft.evaluation.build_hybrid_submission \
  --index_dir "$INDEX" \
  --wan_dir "$BDD_MIX" \
  --region_dir "$WTS_GUARD" \
  --out_dir "$SUBMISSION" \
  --report_csv "$REPORT" \
  --wan_label "cnn_ltx${MIX_LABEL}" \
  --region_label wts_guard015 \
  --region_views wts_overhead_or_fixed_camera wts_vehicle_or_ip_camera \
  --clean

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

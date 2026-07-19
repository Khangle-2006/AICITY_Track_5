#!/usr/bin/env bash
set -euo pipefail

cd /workspace

PYTHON=${PYTHON:-/opt/conda/envs/wan22/bin/python}
INDEX=${INDEX:-/workspace/experiments/index}
TEST_CSV=${TEST_CSV:-$INDEX/test_index.csv}
ROOT=${ROOT:-/workspace/experiments/wan_ticft}
BDD_MIX=${BDD_MIX:-$ROOT/cnn_ltxmix65_bdd_test}
CNN=${CNN:-$ROOT/cnn_ltx_region_selector_bdd_test}
LTX=${LTX:-$ROOT/ltxv_bdd_test_transition16}
WTS_CURRENT=${WTS_CURRENT:-$ROOT/wts_guard_current_components}
WTS_ACTOR=${WTS_ACTOR:-$ROOT/wts_guard_actor_components}
WTS_GUARD=${WTS_GUARD:-$ROOT/wts_guard015_test}
SUBMISSION=${SUBMISSION:-/workspace/experiments/hybrid_cnn_ltxmix65_wts_guard015_candidate}
REPORT=${REPORT:-$ROOT/hybrid_cnn_ltxmix65_wts_guard015_candidate_report.csv}

echo "=== Build CNN ltxmix65 + WTS guard015 candidate ==="

if [[ ! -d "$BDD_MIX" ]]; then
  "$PYTHON" -m wan_ticft.evaluation.materialize_cnn_ltxmix_test \
    --test_csv "$TEST_CSV" \
    --cnn_dir "$CNN" \
    --ltx_dir "$LTX" \
    --out_dir "$BDD_MIX" \
    --ltx_mix 0.65 \
    --clean
fi

echo
echo "[1/5] Generate current WTS route components"
rm -rf "$WTS_CURRENT" "$WTS_ACTOR" "$WTS_GUARD" "$SUBMISSION"
"$PYTHON" -m baseline.cli \
  --index_dir "$INDEX" \
  --out "$WTS_CURRENT" \
  --view_type wts_overhead_or_fixed_camera \
  --method region_norm_medium \
  --region_norm_decay 0.995 \
  --region_norm_max_flow_scale 24.0
"$PYTHON" -m baseline.cli \
  --index_dir "$INDEX" \
  --out "$WTS_CURRENT" \
  --view_type wts_vehicle_or_ip_camera \
  --method region_norm_medium \
  --region_norm_decay 0.96 \
  --region_norm_max_flow_scale 6.0

echo
echo "[2/5] Generate actor-layer overhead candidates with stats"
"$PYTHON" -m baseline.cli \
  --index_dir "$INDEX" \
  --out "$WTS_ACTOR" \
  --view_type wts_overhead_or_fixed_camera \
  --method actor_layer_small_filled \
  --actor_layer_decay 0.985 \
  --actor_layer_max_flow_scale 12.0 \
  --actor_layer_max_area_frac 0.025

echo
echo "[3/5] Materialize guarded WTS route"
"$PYTHON" -m wan_ticft.evaluation.materialize_wts_guarded_test \
  --index_csv "$TEST_CSV" \
  --current_dir "$WTS_CURRENT" \
  --actor_dir "$WTS_ACTOR" \
  --out_dir "$WTS_GUARD" \
  --max_mean 0.018 \
  --max_p90 0.25 \
  --max_motion_p90 6.0 \
  --clean

echo
echo "[4/5] Build hybrid submission directory, no zip"
"$PYTHON" -m wan_ticft.evaluation.build_hybrid_submission \
  --index_dir "$INDEX" \
  --wan_dir "$BDD_MIX" \
  --region_dir "$WTS_GUARD" \
  --out_dir "$SUBMISSION" \
  --report_csv "$REPORT" \
  --wan_label cnn_ltxmix65 \
  --region_label wts_guard015 \
  --region_views wts_overhead_or_fixed_camera wts_vehicle_or_ip_camera \
  --clean

echo
echo "[5/5] Verify full submission folder"
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

#!/usr/bin/env bash
set -euo pipefail

cd /workspace

PYTHON=${PYTHON:-/opt/conda/envs/wan22/bin/python}
INDEX=${INDEX:-/workspace/experiments/index}
TEST_CSV=${TEST_CSV:-$INDEX/test_index.csv}
ROOT=${ROOT:-/workspace/experiments/wan_ticft}
CNN=${CNN:-$ROOT/cnn_ltx_region_selector_bdd_test}
LTX=${LTX:-$ROOT/ltxv_bdd_test_transition16}
BDD_MIX=${BDD_MIX:-$ROOT/cnn_ltxmix65_bdd_test}
WTS=${WTS:-$ROOT/wts_overhead_object_layer_vehicle_medium_test}
SUBMISSION=${SUBMISSION:-/workspace/experiments/hybrid_cnn_ltxmix65_wts_overhead_object_layer_candidate}
REPORT=${REPORT:-$ROOT/hybrid_cnn_ltxmix65_wts_overhead_object_layer_candidate_report.csv}

echo "=== Build CNN ltxmix65 BDD + WTS overhead-object candidate ==="
echo "CNN=${CNN}"
echo "LTX=${LTX}"
echo "BDD_MIX=${BDD_MIX}"
echo "WTS=${WTS}"
echo "SUBMISSION=${SUBMISSION}"

if [[ ! -d "$CNN" ]]; then
  echo "Missing CNN BDD directory: $CNN" >&2
  exit 2
fi
if [[ ! -d "$LTX" ]]; then
  echo "Missing LTX transition16 BDD directory: $LTX" >&2
  exit 2
fi

echo
echo "[1/4] Materialize BDD CNN mixed 65% toward LTX transition16"
"$PYTHON" -m wan_ticft.evaluation.materialize_cnn_ltxmix_test \
  --test_csv "$TEST_CSV" \
  --cnn_dir "$CNN" \
  --ltx_dir "$LTX" \
  --out_dir "$BDD_MIX" \
  --ltx_mix 0.65 \
  --clean

if [[ ! -d "$WTS" ]]; then
  echo
  echo "[2/4] Generate WTS overhead object-layer + vehicle medium"
  "$PYTHON" -m baseline.cli \
    --index_dir "$INDEX" \
    --out "$WTS" \
    --view_type wts_overhead_or_fixed_camera \
    --method actor_layer_small_filled \
    --actor_layer_decay 0.985 \
    --actor_layer_max_flow_scale 12.0 \
    --actor_layer_max_area_frac 0.025

  "$PYTHON" -m baseline.cli \
    --index_dir "$INDEX" \
    --out "$WTS" \
    --view_type wts_vehicle_or_ip_camera \
    --method region_norm_medium \
    --region_norm_decay 0.96 \
    --region_norm_max_flow_scale 6.0
else
  echo
  echo "[2/4] Reuse existing WTS: $WTS"
fi

echo
echo "[3/4] Build hybrid submission directory, no zip"
"$PYTHON" -m wan_ticft.evaluation.build_hybrid_submission \
  --index_dir "$INDEX" \
  --wan_dir "$BDD_MIX" \
  --region_dir "$WTS" \
  --out_dir "$SUBMISSION" \
  --report_csv "$REPORT" \
  --wan_label cnn_ltxmix65 \
  --region_label wts_overhead_object_layer_vehicle_medium \
  --region_views wts_overhead_or_fixed_camera wts_vehicle_or_ip_camera \
  --clean

echo
echo "[4/4] Verify full submission folder"
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

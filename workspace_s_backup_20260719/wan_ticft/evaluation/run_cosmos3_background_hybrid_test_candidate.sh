#!/usr/bin/env bash
set -euo pipefail

echo "=== Build Cosmos3-background BDD + WTS hybrid test candidate ==="
echo "This does not zip. It verifies the folder for robocopy/local zip."

cd /workspace
export PYTHONPATH=/workspace/wan_ticft:${PYTHONPATH:-}

PYTHON=${PYTHON:-/opt/conda/envs/wan22/bin/python}
SERVER=${SERVER:-http://127.0.0.1:8000}
TEST_CSV=${TEST_CSV:-/workspace/experiments/index/test_index.csv}
COSMOS_OUT=${COSMOS_OUT:-/workspace/experiments/wan_ticft/cosmos3_bdd_test_anchor44}
BDD_BASE=${BDD_BASE:-/workspace/experiments/wan_ticft/cnn_ltxmix65_bdd_test}
KERNEL=${KERNEL:-31}
ALPHA=${ALPHA:-0.25}
ALPHA_TAG=${ALPHA_TAG:-$(printf "%03d" "$(python - <<PY
print(round(float("$ALPHA") * 100))
PY
)")}
BDD_OUT=${BDD_OUT:-/workspace/experiments/wan_ticft/cnn_ltxmix65_cosmos_bg_s${KERNEL}_a${ALPHA_TAG}_bdd_test}
WTS_DIR=${WTS_DIR:-/workspace/experiments/wan_ticft/wts_guard015_test}
SUBMIT_OUT=${SUBMIT_OUT:-/workspace/experiments/hybrid_cnn65_cosmos_bg_s${KERNEL}_a${ALPHA_TAG}_wts_guard015_candidate}
REPORT=${REPORT:-/workspace/experiments/wan_ticft/hybrid_cnn65_cosmos_bg_s${KERNEL}_a${ALPHA_TAG}_wts_guard015_candidate_report.csv}

STEPS=${STEPS:-28}
FPS=${FPS:-30}
WIDTH=${WIDTH:-1280}
HEIGHT=${HEIGHT:-720}
GUIDANCE=${GUIDANCE:-8.0}
FLOW_SHIFT=${FLOW_SHIFT:-8.0}
MAX_FRAMES=${MAX_FRAMES:-96}

echo "server=$SERVER"
echo "cosmos_out=$COSMOS_OUT"
echo "bdd_base=$BDD_BASE"
echo "bdd_out=$BDD_OUT"
echo "wts_dir=$WTS_DIR"
echo "submit_out=$SUBMIT_OUT"
echo "kernel=$KERNEL alpha=$ALPHA alpha_tag=$ALPHA_TAG"

"$PYTHON" -m wan_ticft.evaluation.run_cosmos3_bdd_test_anchor \
  --test_csv "$TEST_CSV" \
  --out_dir "$COSMOS_OUT" \
  --server "$SERVER" \
  --width "$WIDTH" \
  --height "$HEIGHT" \
  --fps "$FPS" \
  --steps "$STEPS" \
  --guidance_scale "$GUIDANCE" \
  --flow_shift "$FLOW_SHIFT" \
  --max_frames "$MAX_FRAMES" \
  --resume

"$PYTHON" -m wan_ticft.evaluation.materialize_cosmos_background_hybrid_test \
  --test_csv "$TEST_CSV" \
  --base_dir "$BDD_BASE" \
  --cosmos_report "$COSMOS_OUT/report.csv" \
  --out_dir "$BDD_OUT" \
  --kernel "$KERNEL" \
  --alpha "$ALPHA" \
  --clean

"$PYTHON" -m wan_ticft.evaluation.build_hybrid_submission \
  --index_dir /workspace/experiments/index \
  --wan_dir "$BDD_OUT" \
  --region_dir "$WTS_DIR" \
  --out_dir "$SUBMIT_OUT" \
  --report_csv "$REPORT" \
  --region_views wts_overhead_or_fixed_camera wts_vehicle_or_ip_camera \
  --wan_label "cnn65_cosmos_bg_s${KERNEL}_a${ALPHA_TAG}" \
  --region_label wts_guard015 \
  --clean

"$PYTHON" -m baseline.submission \
  --index_dir /workspace/experiments/index \
  --gen_dir "$SUBMIT_OUT" \
  --submission_dir "$SUBMIT_OUT" \
  --verify_only

cat <<EOF

Candidate verified:
  $SUBMIT_OUT

Robocopy from Windows:
  robocopy S:\\experiments\\$(basename "$SUBMIT_OUT") G:\\$(basename "$SUBMIT_OUT") /E /MT:16 /R:2 /W:2

Notes:
  - Cosmos generation resumes from $COSMOS_OUT/report.csv.
  - If interrupted, rerun the same command; completed BDD rows are skipped.
EOF

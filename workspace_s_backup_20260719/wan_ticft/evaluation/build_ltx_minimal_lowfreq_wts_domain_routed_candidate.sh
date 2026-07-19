#!/usr/bin/env bash
set -euo pipefail

cd /workspace

PYTHON=${PYTHON:-/opt/conda/envs/wan22/bin/python}
INDEX=${INDEX:-/workspace/experiments/index}
TEST_CSV=${TEST_CSV:-$INDEX/test_index.csv}
ROOT=${ROOT:-/workspace/experiments/wan_ticft}
BDD_MINIMAL=${BDD_MINIMAL:-$ROOT/ltxv_bdd_test_minimal_prompt_full}
BDD_MINIMAL_LOWFREQ=${BDD_MINIMAL_LOWFREQ:-$ROOT/ltxv_bdd_test_minimal_lowfreq_s16_a50}
WTS=${WTS:-$ROOT/wts_overhead_long24_vehicle_medium_test_minimal_lowfreq}
SUBMISSION=${SUBMISSION:-/workspace/experiments/hybrid_ltx_minimal_lowfreq_wts_domain_routed_candidate}
REPORT=${REPORT:-/workspace/experiments/wan_ticft/hybrid_ltx_minimal_lowfreq_wts_domain_routed_candidate_report.csv}
GPU_ID="${CUDA_VISIBLE_DEVICES:-3}"
GPU_ID="${GPU_ID%%,*}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-3}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

echo "=== Build BDD minimal prompt + lowfreq, WTS current routed candidate ==="
echo "BDD_MINIMAL=${BDD_MINIMAL}"
echo "BDD_MINIMAL_LOWFREQ=${BDD_MINIMAL_LOWFREQ}"
echo "WTS=${WTS}"
echo "SUBMISSION=${SUBMISSION}"

echo
echo "[1/6] Generate BDD test with minimal LTX prompt"
/workspace/wan_ticft/evaluation/check_gpu_free.sh "$GPU_ID" 24000
"$PYTHON" -m wan_ticft.evaluation.run_ltxv_test_bdd \
  --test_csv "$TEST_CSV" \
  --out_dir "$BDD_MINIMAL" \
  --ltx_repo /workspace/third_party/LTX-Video \
  --ltx_python "$ROOT/ltxv_env/bin/python" \
  --pipeline_config /workspace/wan_ticft/evaluation/ltxv_2b_gate.yaml \
  --prompt_mode minimal \
  --resume

echo
echo "[2/6] Apply low-frequency region correction to BDD minimal outputs"
"$PYTHON" -m wan_ticft.evaluation.materialize_ltx_region_lowfreq_test \
  --test_csv "$TEST_CSV" \
  --ltx_dir "$BDD_MINIMAL" \
  --out_dir "$BDD_MINIMAL_LOWFREQ" \
  --sigma 16 \
  --alpha 0.50 \
  --clean

echo
echo "[3/6] Generate WTS overhead/fixed current route: long24"
rm -rf "$WTS" "$SUBMISSION"
"$PYTHON" -m baseline.cli \
  --index_dir "$INDEX" \
  --out "$WTS" \
  --view_type wts_overhead_or_fixed_camera \
  --method region_norm_medium \
  --region_norm_decay 0.995 \
  --region_norm_max_flow_scale 24.0

echo
echo "[4/6] Generate WTS vehicle/IP current route: medium"
"$PYTHON" -m baseline.cli \
  --index_dir "$INDEX" \
  --out "$WTS" \
  --view_type wts_vehicle_or_ip_camera \
  --method region_norm_medium \
  --region_norm_decay 0.96 \
  --region_norm_max_flow_scale 6.0

echo
echo "[5/6] Build hybrid submission directory, no zip"
"$PYTHON" -m wan_ticft.evaluation.build_hybrid_submission \
  --index_dir "$INDEX" \
  --wan_dir "$BDD_MINIMAL_LOWFREQ" \
  --region_dir "$WTS" \
  --out_dir "$SUBMISSION" \
  --report_csv "$REPORT" \
  --wan_label ltx_minimal_lowfreq_s16_a50 \
  --region_label wts_overhead_long24_vehicle_medium \
  --region_views wts_overhead_or_fixed_camera wts_vehicle_or_ip_camera \
  --clean

echo
echo "[6/6] Verify full submission folder"
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

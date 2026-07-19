#!/usr/bin/env bash
set -euo pipefail

cd /workspace

PYTHON=${PYTHON:-/opt/conda/envs/wan22/bin/python}
INDEX_DIR=${INDEX_DIR:-/workspace/experiments/index}
BDD_DIR=${BDD_DIR:-/workspace/experiments/wan_ticft/ltxv_bdd_test_transition16}
WTS_DIR=${WTS_DIR:-/workspace/experiments/wan_ticft/wts_overhead_long24_vehicle_guard25_test}
SUBMISSION_DIR=${SUBMISSION_DIR:-/workspace/experiments/hybrid_ltx_transition16_wts_vehicle_guard25_candidate}
REPORT_CSV=${REPORT_CSV:-/workspace/experiments/wan_ticft/hybrid_ltx_transition16_wts_vehicle_guard25_candidate_report.csv}

echo "=== Build BDD LTX transition16 + WTS overhead long24 + vehicle guard25 candidate ==="
echo "BDD_DIR=${BDD_DIR}"
echo "WTS_DIR=${WTS_DIR}"
echo "SUBMISSION_DIR=${SUBMISSION_DIR}"

if [[ ! -d "${BDD_DIR}" ]]; then
  echo "Missing BDD_DIR: ${BDD_DIR}" >&2
  exit 2
fi

rm -rf "${WTS_DIR}" "${SUBMISSION_DIR}"

echo
echo "[1/5] Generate WTS overhead/fixed with validated long24 region flow"
"${PYTHON}" -m baseline.cli \
  --index_dir "${INDEX_DIR}" \
  --out "${WTS_DIR}" \
  --view_type wts_overhead_or_fixed_camera \
  --method region_norm_medium \
  --region_norm_decay 0.995 \
  --region_norm_max_flow_scale 24.0

echo
echo "[2/5] Generate WTS vehicle/IP with guarded medium region flow"
"${PYTHON}" -m baseline.cli \
  --index_dir "${INDEX_DIR}" \
  --out "${WTS_DIR}" \
  --view_type wts_vehicle_or_ip_camera \
  --method region_norm_medium \
  --region_norm_decay 0.96 \
  --region_norm_max_flow_scale 6.0 \
  --region_actor_guard_strength 0.25

echo
echo "[3/5] Build hybrid submission directory, no zip"
"${PYTHON}" -m wan_ticft.evaluation.build_hybrid_submission \
  --index_dir "${INDEX_DIR}" \
  --wan_dir "${BDD_DIR}" \
  --region_dir "${WTS_DIR}" \
  --out_dir "${SUBMISSION_DIR}" \
  --report_csv "${REPORT_CSV}" \
  --wan_label ltx_transition16 \
  --region_label wts_overhead_long24_vehicle_guard25 \
  --region_views wts_overhead_or_fixed_camera wts_vehicle_or_ip_camera \
  --clean

echo
echo "[4/5] Verify full submission folder"
"${PYTHON}" -m baseline.submission \
  --index_dir "${INDEX_DIR}" \
  --gen_dir "${SUBMISSION_DIR}" \
  --submission_dir "${SUBMISSION_DIR}" \
  --verify_only

echo
echo "[5/5] Build WTS contact sheets for visual QA"
"${PYTHON}" -m baseline.video_contact_sheet \
  --input "${WTS_DIR}" \
  --out_dir "${WTS_DIR}_sheets" \
  --mode frames \
  --samples 16

echo
echo "Candidate verified but not zipped:"
echo "  ${SUBMISSION_DIR}"
echo "Report:"
echo "  ${REPORT_CSV}"
echo "WTS sheets:"
echo "  ${WTS_DIR}_sheets/index.html"

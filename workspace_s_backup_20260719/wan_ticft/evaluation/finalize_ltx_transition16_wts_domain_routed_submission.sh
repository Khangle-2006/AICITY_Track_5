#!/usr/bin/env bash
set -euo pipefail

cd /workspace

PYTHON=/opt/conda/envs/wan22/bin/python
INDEX=/workspace/experiments/index
CANDIDATE=/workspace/experiments/hybrid_ltx_transition16_wts_domain_routed_candidate
VERIFIED=/workspace/experiments/hybrid_ltx_transition16_wts_domain_routed_submission_verified
REPORT=/workspace/experiments/hybrid_ltx_transition16_wts_domain_routed_submission_verify.csv

"$PYTHON" -m baseline.submission \
  --index_dir "$INDEX" \
  --gen_dir "$CANDIDATE" \
  --submission_dir "$VERIFIED" \
  --report_csv "$REPORT" \
  --clean

echo
echo "Verified submission directory:"
echo "  $VERIFIED"
echo "No ZIP was created. Copy this directory to local and zip it there."

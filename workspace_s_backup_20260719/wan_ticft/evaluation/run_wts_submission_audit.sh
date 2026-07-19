#!/usr/bin/env bash
set -euo pipefail

cd /workspace

AUDIT=/workspace/experiments/wan_ticft/region_norm_medium_wts_audit

/opt/conda/envs/wan22/bin/python -m wan_ticft.evaluation.prepare_wts_submission_audit \
  --test_csv /workspace/experiments/index/test_index.csv \
  --source_dir /workspace/experiments/region_norm_medium_submission \
  --out_dir "$AUDIT" \
  --clean

/opt/conda/envs/wan22/bin/python -m baseline.video_contact_sheet \
  --input "$AUDIT" \
  --mode frames \
  --out_dir "${AUDIT}_sheets" \
  --samples 16 \
  --tile_width 320 \
  --cols 4

/opt/conda/envs/wan22/bin/python -m baseline.diagnose_motion \
  --index_dir /workspace/experiments/index \
  --gen_dir /workspace/experiments/region_norm_medium_submission \
  --out_csv /workspace/experiments/wan_ticft/region_norm_medium_wts_motion_audit.csv

echo "Inspect: ${AUDIT}_sheets"

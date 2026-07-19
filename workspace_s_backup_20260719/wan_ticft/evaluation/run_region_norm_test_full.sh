#!/usr/bin/env bash
set -euo pipefail

cd /workspace

python -m baseline.cli \
  --index_dir /workspace/experiments/index \
  --out /workspace/experiments/region_norm_medium_test \
  --method region_norm_medium

python -m baseline.submission \
  --index_dir /workspace/experiments/index \
  --gen_dir /workspace/experiments/region_norm_medium_test \
  --submission_dir /workspace/experiments/region_norm_medium_submission \
  --report_csv /workspace/experiments/region_norm_medium_submission_verify.csv

python -m baseline.video_contact_sheet \
  --input /workspace/experiments/region_norm_medium_test \
  --mode frames \
  --out_dir /workspace/experiments/region_norm_medium_test_sheets \
  --samples 12 \
  --tile_width 320 \
  --cols 4

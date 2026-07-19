#!/usr/bin/env bash
set -euo pipefail

cd /workspace

python -m baseline.submission \
  --index_dir /workspace/experiments/index \
  --gen_dir /workspace/experiments/region_norm_medium_test \
  --submission_dir /workspace/experiments/region_norm_medium_submission \
  --report_csv /workspace/experiments/region_norm_medium_submission_verify.csv \
  --zip /workspace/experiments/region_norm_medium_submission_flat.zip

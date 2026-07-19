#!/usr/bin/env bash
set -euo pipefail

cd /workspace

python -m baseline.cli \
  --index_dir /workspace/experiments/index \
  --out /workspace/experiments/region_norm_medium_test_smoke4 \
  --method region_norm_medium \
  --limit 4 \
  --make_montage

python -m baseline.video_contact_sheet \
  --input /workspace/experiments/region_norm_medium_test_smoke4 \
  --mode frames \
  --out_dir /workspace/experiments/region_norm_medium_test_smoke4_sheets \
  --samples 12 \
  --tile_width 320 \
  --cols 4

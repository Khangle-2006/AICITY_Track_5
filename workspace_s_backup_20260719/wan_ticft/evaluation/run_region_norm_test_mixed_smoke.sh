#!/usr/bin/env bash
set -euo pipefail

cd /workspace

for view in bdd wts_overhead_or_fixed_camera wts_vehicle_or_ip_camera; do
  out="/workspace/experiments/region_norm_medium_test_smoke_${view}"
  python -m baseline.cli \
    --index_dir /workspace/experiments/index \
    --out "$out" \
    --method region_norm_medium \
    --view_type "$view" \
    --limit 2 \
    --make_montage

  python -m baseline.video_contact_sheet \
    --input "$out" \
    --mode frames \
    --out_dir "${out}_sheets" \
    --samples 12 \
    --tile_width 320 \
    --cols 4
done

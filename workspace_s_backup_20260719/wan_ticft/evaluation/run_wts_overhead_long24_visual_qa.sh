#!/usr/bin/env bash
set -euo pipefail

cd /workspace

PYTHON=/opt/conda/envs/wan22/bin/python
ROOT=/workspace/experiments/wan_ticft
OUT="$ROOT/wts_distribution_gate20"
QA="$ROOT/wts_overhead_long24_visual_qa"

"$PYTHON" -m wan_ticft.evaluation.prepare_report_video_subset \
  --view_type wts_overhead_or_fixed_camera \
  --report "target=$OUT/target_report.csv" \
  --report "region_medium=$OUT/region_norm_medium_report.csv" \
  --report "region_long24=$OUT/region_norm_long24_report.csv" \
  --out_dir "$QA" \
  --clean

for LABEL in target region_medium region_long24; do
  "$PYTHON" -m baseline.video_contact_sheet \
    --input "$QA/$LABEL" \
    --mode videos \
    --out_dir "${QA}_${LABEL}_sheets" \
    --samples 16 \
    --tile_width 320 \
    --cols 4
done

echo
echo "Inspect the same sample IDs across:"
echo "  ${QA}_target_sheets/index.html"
echo "  ${QA}_region_medium_sheets/index.html"
echo "  ${QA}_region_long24_sheets/index.html"
echo "Reject long24 for obvious stretching, duplicated actors, or edge ghosting."

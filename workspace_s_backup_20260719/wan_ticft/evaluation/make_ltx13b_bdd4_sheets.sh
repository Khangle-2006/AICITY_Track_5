#!/usr/bin/env bash
set -euo pipefail

cd /workspace

PYTHON=${PYTHON:-/opt/conda/envs/wan22/bin/python}
ROOT=${ROOT:-/workspace/experiments/wan_ticft}
OUT=${OUT:-$ROOT/ltx13b_traffic_bdd4_gate}
DEPTH_REPORT=${DEPTH_REPORT:-$ROOT/bdd_supervised_ltx_depth_eval24/depth/depth_layered_a75/report.csv}

mapfile -t SAMPLE_IDS < <("$PYTHON" - <<PY
import json
for line in open("$OUT/selected_manifest.jsonl", encoding="utf-8"):
    if line.strip():
        print(json.loads(line)["sample_id"])
PY
)

for sample_id in "${SAMPLE_IDS[@]}"; do
  "$PYTHON" -m wan_ticft.evaluation.make_report_comparison_sheet \
    --manifest "$OUT/selected_manifest.jsonl" \
    --reports ltx13b="$OUT/report.csv" depth_a75="$DEPTH_REPORT" \
    --sample_id "$sample_id" \
    --out_dir "$OUT/compare_sheets" \
    --frames 8 \
    --tile_width 220
done

echo "sheets=$OUT/compare_sheets"

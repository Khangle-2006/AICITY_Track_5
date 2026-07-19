#!/usr/bin/env bash
set -euo pipefail

cd /workspace

PYTHON=${PYTHON:-/opt/conda/envs/wan22/bin/python}
ROOT=${ROOT:-/workspace/experiments/wan_ticft}
OUT=${OUT:-$ROOT/wts_object_layer_small_gate20}
ROUTE_DIR=${ROUTE_DIR:-$OUT/routed_reports}

echo "=== WTS object-layer small correction pixel gate ==="

"$PYTHON" -m wan_ticft.evaluation.evaluate_video_report_pixels \
  --target_report "$OUT/target_report.csv" \
  --report "current=$ROUTE_DIR/current_report.csv" \
  --report "vehicle_filled=$ROUTE_DIR/vehicle_actor_layer_filled_report.csv" \
  --report "vehicle_small_filled=$ROUTE_DIR/vehicle_actor_layer_small_filled_report.csv" \
  --report "overhead_small_filled=$ROUTE_DIR/overhead_actor_layer_small_filled_report.csv" \
  --out_csv "$ROOT/wts_object_layer_small_pixel_gate20.csv"

echo
echo "Decision note:"
echo "  - vehicle_small_filled already fails FVD proxy badly."
echo "  - Only consider submitting it if PSNR/SSIM gain is large enough to justify"
echo "    a likely FVD hit. Otherwise keep the overhead-only candidate."

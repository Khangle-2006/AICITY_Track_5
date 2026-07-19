#!/usr/bin/env bash
set -euo pipefail

cd /workspace

PYTHON=${PYTHON:-/opt/conda/envs/wan22/bin/python}
ROOT=${ROOT:-/workspace/experiments/wan_ticft}
OUT=${OUT:-$ROOT/wts_object_layer_small_gate20}
ROUTE_DIR=${ROUTE_DIR:-$OUT/routed_reports}
MANIFEST=${MANIFEST:-$ROOT/manifests/val_ticft_balanced_125_frames.jsonl}
GPU_ID="${CUDA_VISIBLE_DEVICES:-3}"
GPU_ID="${GPU_ID%%,*}"

echo "=== WTS object-layer small-blob correction gate ==="
echo "Correction: object-layer fill ignores large foreground/reflection blobs."

rm -rf "$OUT"
mkdir -p "$ROUTE_DIR"

"$PYTHON" -m wan_ticft.evaluation.materialize_wts_distribution_gate \
  --manifest "$MANIFEST" \
  --out_dir "$OUT" \
  --limit_per_domain 10 \
  --frames 50 \
  --candidates \
    region_norm_medium \
    region_norm_long24 \
    actor_layer_filled \
    actor_layer_small_filled \
  --clean

"$PYTHON" -m wan_ticft.evaluation.build_wts_view_routed_report \
  --overhead_report "$OUT/region_norm_long24_report.csv" \
  --vehicle_report "$OUT/region_norm_medium_report.csv" \
  --route_label current_overhead_long24_vehicle_medium \
  --out_report "$ROUTE_DIR/current_report.csv"
"$PYTHON" -m wan_ticft.evaluation.build_wts_view_routed_report \
  --overhead_report "$OUT/region_norm_long24_report.csv" \
  --vehicle_report "$OUT/actor_layer_filled_report.csv" \
  --route_label vehicle_actor_layer_filled \
  --out_report "$ROUTE_DIR/vehicle_actor_layer_filled_report.csv"
"$PYTHON" -m wan_ticft.evaluation.build_wts_view_routed_report \
  --overhead_report "$OUT/region_norm_long24_report.csv" \
  --vehicle_report "$OUT/actor_layer_small_filled_report.csv" \
  --route_label vehicle_actor_layer_small_filled \
  --out_report "$ROUTE_DIR/vehicle_actor_layer_small_filled_report.csv"
"$PYTHON" -m wan_ticft.evaluation.build_wts_view_routed_report \
  --overhead_report "$OUT/actor_layer_small_filled_report.csv" \
  --vehicle_report "$OUT/region_norm_medium_report.csv" \
  --route_label overhead_actor_layer_small_filled \
  --out_report "$ROUTE_DIR/overhead_actor_layer_small_filled_report.csv"

echo
echo "=== Relative FID/FVD proxy ==="
/workspace/wan_ticft/evaluation/check_gpu_free.sh "$GPU_ID" 3000
CUDA_VISIBLE_DEVICES="$GPU_ID" \
TORCH_HOME="$ROOT/tools/torch_cache" \
"$PYTHON" -m wan_ticft.evaluation.audit_perceptual_video_distribution \
  --manifest "$MANIFEST" \
  --target_report "$OUT/target_report.csv" \
  --full_video \
  --report "current=$ROUTE_DIR/current_report.csv" \
  --report "vehicle_filled=$ROUTE_DIR/vehicle_actor_layer_filled_report.csv" \
  --report "vehicle_small_filled=$ROUTE_DIR/vehicle_actor_layer_small_filled_report.csv" \
  --report "overhead_small_filled=$ROUTE_DIR/overhead_actor_layer_small_filled_report.csv" \
  --out_csv "$ROOT/wts_object_layer_small_routed_proxy_gate20.csv"

echo
echo "=== Contact sheets ==="
"$PYTHON" -m baseline.video_contact_sheet \
  --input "$OUT/actor_layer_small_filled" \
  --mode videos \
  --out_dir "$OUT/actor_layer_small_filled_sheets" \
  --samples 16 \
  --tile_width 320 \
  --cols 4

echo
echo "Correction rule:"
echo "  - Continue only if small-filled still improves both FID and FVD versus current"
echo "    and visual QA removes the large cut-paste/reflection artifacts."
echo "  - If small-filled loses the proxy advantage or still has halos, reject object-layer."

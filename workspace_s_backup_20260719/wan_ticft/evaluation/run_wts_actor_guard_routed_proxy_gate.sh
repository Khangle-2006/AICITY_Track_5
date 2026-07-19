#!/usr/bin/env bash
set -euo pipefail

cd /workspace

PYTHON=/opt/conda/envs/wan22/bin/python
ROOT=/workspace/experiments/wan_ticft
OUT="$ROOT/wts_actor_guard_gate20"
MANIFEST="$ROOT/manifests/val_ticft_balanced_125_frames.jsonl"
ROUTE_DIR="$OUT/routed_reports"
GPU_ID="${CUDA_VISIBLE_DEVICES:-3}"
GPU_ID="${GPU_ID%%,*}"

mkdir -p "$ROUTE_DIR"

build_route() {
  local label="$1"
  local vehicle_report="$2"
  "$PYTHON" -m wan_ticft.evaluation.build_wts_view_routed_report \
    --overhead_report "$OUT/region_norm_long24_report.csv" \
    --vehicle_report "$vehicle_report" \
    --route_label "$label" \
    --out_report "$ROUTE_DIR/${label}_report.csv"
}

build_route current_overhead_long24_vehicle_medium "$OUT/region_norm_medium_report.csv"
build_route vehicle_guard25 "$OUT/region_actor_guard25_report.csv"
build_route vehicle_guard50 "$OUT/region_actor_guard50_report.csv"
build_route vehicle_long24_guard50 "$OUT/region_long24_actor_guard50_report.csv"

/workspace/wan_ticft/evaluation/check_gpu_free.sh "$GPU_ID" 3000
CUDA_VISIBLE_DEVICES="$GPU_ID" \
TORCH_HOME="$ROOT/tools/torch_cache" \
"$PYTHON" -m wan_ticft.evaluation.audit_perceptual_video_distribution \
  --manifest "$MANIFEST" \
  --target_report "$OUT/target_report.csv" \
  --full_video \
  --report "current=$ROUTE_DIR/current_overhead_long24_vehicle_medium_report.csv" \
  --report "vehicle_guard25=$ROUTE_DIR/vehicle_guard25_report.csv" \
  --report "vehicle_guard50=$ROUTE_DIR/vehicle_guard50_report.csv" \
  --report "vehicle_long24_guard50=$ROUTE_DIR/vehicle_long24_guard50_report.csv" \
  --out_csv "$ROOT/wts_actor_guard_routed_proxy_gate20.csv"

echo
echo "Routing rule:"
echo "  - Overhead stays region_long24 because actor guard worsens overhead FVD."
echo "  - Promote a vehicle/IP guard route only if aggregate FID and FVD improve"
echo "    versus current, and visual QA shows less smear without freeze/halos."

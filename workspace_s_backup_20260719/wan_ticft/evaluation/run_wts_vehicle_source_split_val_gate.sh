#!/usr/bin/env bash
set -euo pipefail

cd /workspace

PYTHON=${PYTHON:-/opt/conda/envs/wan22/bin/python}
ROOT=${ROOT:-/workspace/experiments/wan_ticft}
MANIFEST=${MANIFEST:-$ROOT/manifests/val_ticft_balanced_125_frames.jsonl}
OUT=${OUT:-$ROOT/wts_vehicle_source_split_val24}
LIMIT_PER_SPLIT=${LIMIT_PER_SPLIT:-12}
MODEL_CACHE=${MODEL_CACHE:-$ROOT/depth_layered_models}
GPU_ID="${CUDA_VISIBLE_DEVICES:-3}"
GPU_ID="${GPU_ID%%,*}"

SELECTED="$OUT/selected_vehicle_source_split_manifest.jsonl"
BASE_OUT="$OUT/base"
DEPTH_OUT="$OUT/depth"
ROUTE_DIR="$OUT/routed_reports"

echo "=== WTS vehicle/IP source split validation gate ==="
echo "out=$OUT"
echo "limit_per_split=$LIMIT_PER_SPLIT"

rm -rf "$OUT"
mkdir -p "$ROUTE_DIR"

echo
echo "[1/6] Select balanced vehicle/IP validation manifest"
"$PYTHON" -m wan_ticft.evaluation.select_wts_vehicle_source_split_manifest \
  --manifest "$MANIFEST" \
  --out_manifest "$SELECTED" \
  --limit_per_split "$LIMIT_PER_SPLIT"

echo
echo "[2/6] Materialize current WTS vehicle base and target"
"$PYTHON" -m wan_ticft.evaluation.materialize_wts_distribution_gate \
  --manifest "$SELECTED" \
  --out_dir "$BASE_OUT" \
  --limit_per_domain "$((LIMIT_PER_SPLIT * 2))" \
  --views wts_vehicle_or_ip_camera \
  --frames 50 \
  --cover_manifest_targets \
  --candidates region_norm_medium \
  --clean

echo
echo "[3/6] Render depth-a75 over vehicle base"
/workspace/wan_ticft/evaluation/check_gpu_free.sh "$GPU_ID" 12000
CUDA_VISIBLE_DEVICES="$GPU_ID" \
HF_HOME="$MODEL_CACHE/hf_home" \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
"$PYTHON" -m wan_ticft.evaluation.materialize_depth_layered_renderer \
  --manifest "$SELECTED" \
  --base_report "$BASE_OUT/region_norm_medium_report.csv" \
  --out_root "$DEPTH_OUT" \
  --model_cache "$MODEL_CACHE/huggingface" \
  --limit "$((LIMIT_PER_SPLIT * 2))" \
  --view_type wts_vehicle_or_ip_camera \
  --preserve_full_video \
  --alphas 0.75 \
  --focal_scale 0.9 \
  --point_stride 2 \
  --splat_radius 1 \
  --min_matches 20 \
  --min_inliers 12 \
  --photo_sigma 32 \
  --segment_batch 4 \
  --pose_smoothing_window 7 \
  --filled_pose_scale 0.35 \
  --confidence_ema 0.65 \
  --clean

echo
echo "[4/6] Build routed validation reports"
for ROUTE in current normal_only event_only all_depth; do
  "$PYTHON" -m wan_ticft.evaluation.build_wts_vehicle_source_split_report \
    --manifest "$SELECTED" \
    --base_report "$BASE_OUT/region_norm_medium_report.csv" \
    --depth_report "$DEPTH_OUT/depth_layered_a75/report.csv" \
    --route "$ROUTE" \
    --out_report "$ROUTE_DIR/${ROUTE}_report.csv"
done

echo
echo "[5/6] Pixel metrics"
"$PYTHON" -m wan_ticft.evaluation.evaluate_video_report_pixels \
  --target_report "$BASE_OUT/target_report.csv" \
  --report "current=$ROUTE_DIR/current_report.csv" \
  --report "normal_only=$ROUTE_DIR/normal_only_report.csv" \
  --report "event_only=$ROUTE_DIR/event_only_report.csv" \
  --report "all_depth=$ROUTE_DIR/all_depth_report.csv" \
  --out_csv "$ROOT/wts_vehicle_source_split_val${LIMIT_PER_SPLIT}x2_pixel.csv"

echo
echo "[6/6] FID/FVD proxy"
/workspace/wan_ticft/evaluation/check_gpu_free.sh "$GPU_ID" 3000
CUDA_VISIBLE_DEVICES="$GPU_ID" \
TORCH_HOME="$ROOT/tools/torch_cache" \
"$PYTHON" -m wan_ticft.evaluation.audit_perceptual_video_distribution \
  --manifest "$SELECTED" \
  --target_report "$BASE_OUT/target_report.csv" \
  --full_video \
  --report "current=$ROUTE_DIR/current_report.csv" \
  --report "normal_only=$ROUTE_DIR/normal_only_report.csv" \
  --report "event_only=$ROUTE_DIR/event_only_report.csv" \
  --report "all_depth=$ROUTE_DIR/all_depth_report.csv" \
  --out_csv "$ROOT/wts_vehicle_source_split_val${LIMIT_PER_SPLIT}x2_proxy.csv"

echo
echo "Gate rule:"
echo "  - Submit/keep normal_only only if it improves FID/FVD proxy versus current"
echo "    without a material pixel collapse."
echo "  - If normal_only fails and event_only passes, build/submit event route next."
echo "  - If both fail, keep the original WTS base route."
echo
echo "pixel=$ROOT/wts_vehicle_source_split_val${LIMIT_PER_SPLIT}x2_pixel.csv"
echo "proxy=$ROOT/wts_vehicle_source_split_val${LIMIT_PER_SPLIT}x2_proxy.csv"
echo "reports=$ROUTE_DIR"

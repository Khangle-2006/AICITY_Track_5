#!/usr/bin/env bash
set -euo pipefail

cd /workspace

PYTHON=/opt/conda/envs/wan22/bin/python
INDEX=/workspace/experiments/index
TEST_CSV="$INDEX/test_index.csv"
ROOT=/workspace/experiments/wan_ticft
BDD_RENDERED="$ROOT/depth_layered_a75_bdd_test"
WTS_BASE="$ROOT/wts_overhead_object_layer_vehicle_medium_test"
WTS_VEHICLE_RENDERED="$ROOT/depth_layered_a75_wts_vehicle_test"
MODEL_CACHE="$ROOT/depth_layered_models"
SUBMISSION=/workspace/experiments/hybrid_depth_a75_bdd_wts_vehicle_overhead_object_candidate
REPORT="$ROOT/hybrid_depth_a75_bdd_wts_vehicle_overhead_object_candidate_report.csv"
GPU_ID="${CUDA_VISIBLE_DEVICES:-3}"
GPU_ID="${GPU_ID%%,*}"

echo "=== Build BDD depth-a75 + WTS vehicle depth-a75 + unchanged WTS overhead candidate ==="
echo "No zip is created."
for path in "$TEST_CSV" "$BDD_RENDERED/report.csv"; do
  [[ -f "$path" ]] || { echo "Missing input: $path" >&2; exit 2; }
done
[[ -d "$WTS_BASE" ]] || { echo "Missing WTS base: $WTS_BASE" >&2; exit 2; }
/workspace/wan_ticft/evaluation/check_gpu_free.sh "$GPU_ID" 12000

echo "[1/3] Apply validated depth-a75 renderer to 20 WTS vehicle/IP test samples"
CUDA_VISIBLE_DEVICES="$GPU_ID" \
HF_HOME="$MODEL_CACHE/hf_home" \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
"$PYTHON" -m wan_ticft.evaluation.materialize_depth_layered_renderer_test \
  --test_csv "$TEST_CSV" \
  --base_dir "$WTS_BASE" \
  --out_dir "$WTS_VEHICLE_RENDERED" \
  --model_cache "$MODEL_CACHE/huggingface" \
  --view_type wts_vehicle_or_ip_camera \
  --alpha 0.75 \
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

echo "[2/3] Route BDD, WTS overhead, and WTS vehicle sources"
"$PYTHON" -m wan_ticft.evaluation.build_hybrid_submission \
  --index_dir "$INDEX" \
  --wan_dir "$BDD_RENDERED" \
  --region_dir "$WTS_BASE" \
  --out_dir "$SUBMISSION" \
  --report_csv "$REPORT" \
  --wan_label depth_layered_a75_bdd \
  --region_label wts_overhead_object_layer \
  --region_views wts_overhead_or_fixed_camera wts_vehicle_or_ip_camera \
  --view_override "wts_vehicle_or_ip_camera=$WTS_VEHICLE_RENDERED=depth_layered_a75_wts_vehicle" \
  --clean

echo "[3/3] Verify complete 71-sample submission folder"
"$PYTHON" -m baseline.submission \
  --index_dir "$INDEX" \
  --gen_dir "$SUBMISSION" \
  --submission_dir "$SUBMISSION" \
  --verify_only

echo
echo "Candidate verified but not zipped:"
echo "  $SUBMISSION"
echo "Report:"
echo "  $REPORT"
echo "WTS vehicle diagnostics:"
echo "  $WTS_VEHICLE_RENDERED/diagnostics.csv"

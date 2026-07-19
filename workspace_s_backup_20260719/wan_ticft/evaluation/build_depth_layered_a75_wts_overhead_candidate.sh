#!/usr/bin/env bash
set -euo pipefail

cd /workspace

PYTHON=/opt/conda/envs/wan22/bin/python
INDEX=/workspace/experiments/index
TEST_CSV="$INDEX/test_index.csv"
ROOT=/workspace/experiments/wan_ticft
BDD_BASE="$ROOT/cnn_ltxmix65_bdd_test"
BDD_RENDERED="$ROOT/depth_layered_a75_bdd_test"
MODEL_CACHE="$ROOT/depth_layered_models"
WTS="$ROOT/wts_overhead_object_layer_vehicle_medium_test"
SUBMISSION=/workspace/experiments/hybrid_depth_layered_a75_wts_overhead_object_layer_candidate
REPORT="$ROOT/hybrid_depth_layered_a75_wts_overhead_object_layer_candidate_report.csv"
GPU_ID="${CUDA_VISIBLE_DEVICES:-3}"
GPU_ID="${GPU_ID%%,*}"

echo "=== Build depth-layered a75 BDD + current WTS candidate ==="
echo "No zip is created."
for path in "$TEST_CSV" "$BDD_BASE/report.csv"; do
  if [[ ! -f "$path" ]]; then
    echo "Missing input: $path" >&2
    exit 2
  fi
done
if [[ ! -d "$WTS" ]]; then
  echo "Missing current WTS source: $WTS" >&2
  exit 2
fi
/workspace/wan_ticft/evaluation/check_gpu_free.sh "$GPU_ID" 12000

echo "[1/3] Render BDD test with validated temporal depth-layered a75"
CUDA_VISIBLE_DEVICES="$GPU_ID" \
HF_HOME="$MODEL_CACHE/hf_home" \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
"$PYTHON" -m wan_ticft.evaluation.materialize_depth_layered_renderer_test \
  --test_csv "$TEST_CSV" \
  --base_dir "$BDD_BASE" \
  --out_dir "$BDD_RENDERED" \
  --model_cache "$MODEL_CACHE/huggingface" \
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

echo "[2/3] Build hybrid folder with unchanged WTS/object-layer outputs"
"$PYTHON" -m wan_ticft.evaluation.build_hybrid_submission \
  --index_dir "$INDEX" \
  --wan_dir "$BDD_RENDERED" \
  --region_dir "$WTS" \
  --out_dir "$SUBMISSION" \
  --report_csv "$REPORT" \
  --wan_label depth_layered_a75 \
  --region_label wts_overhead_object_layer_vehicle_medium \
  --region_views wts_overhead_or_fixed_camera wts_vehicle_or_ip_camera \
  --clean

echo "[3/3] Verify complete submission folder"
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
echo "BDD diagnostics:"
echo "  $BDD_RENDERED/diagnostics.csv"

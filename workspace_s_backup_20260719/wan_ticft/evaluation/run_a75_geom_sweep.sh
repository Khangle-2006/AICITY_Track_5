#!/bin/bash
set -euo pipefail
# Pareto attempt for a75: sweep splat_radius/point_stride to reduce disocclusion DRAG
# (stretched warp) without hurting metrics. Measures REAL-FID + PSNR/SSIM on val (BDD24).
cd /workspace
PYTHON=${PYTHON:-/opt/conda/envs/wan22/bin/python}
ROOT=/workspace/experiments/wan_ticft
MANIFEST=$ROOT/ltxv_bdd25_gate/selected_manifest.jsonl
BASE_REPORT=$ROOT/ltxv_bdd25_gate/report.csv
MODEL_CACHE=$ROOT/depth_layered_models/huggingface
REAL=$ROOT/frames_balanced_val_125/val
OUT=$ROOT/a75_geom_sweep
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2}"

render() {  # tag radius stride
  local tag=$1 r=$2 s=$3
  echo ">>> render $tag (splat_radius=$r point_stride=$s)"
  "$PYTHON" -m wan_ticft.evaluation.materialize_depth_layered_renderer \
    --manifest "$MANIFEST" --base_report "$BASE_REPORT" --out_root "$OUT/$tag" \
    --model_cache "$MODEL_CACHE" --view_type bdd --limit 24 --preserve_full_video \
    --alphas 0.75 --focal_scale 0.9 --point_stride "$s" --splat_radius "$r" \
    --min_matches 20 --min_inliers 12 --photo_sigma 32 --segment_batch 4 \
    --pose_smoothing_window 7 --filled_pose_scale 0.35 --confidence_ema 0.65
}

# configs: baseline r1s2 (=a75), then drag-reduction variants
render r1s2 1 2
render r0s1 0 1
render r1s1 1 1
render r0s2 0 2

echo "=== PIXEL metrics (vs GT) ==="
"$PYTHON" -m wan_ticft.evaluation.evaluate_video_reports --manifest "$MANIFEST" \
  --reports r1s2="$OUT/r1s2/depth_layered_a75/report.csv" \
            r0s1="$OUT/r0s1/depth_layered_a75/report.csv" \
            r1s1="$OUT/r1s1/depth_layered_a75/report.csv" \
            r0s2="$OUT/r0s2/depth_layered_a75/report.csv" \
  --offset_mode timestamp --out_csv "$OUT/eval_pixel.csv"

echo "=== REAL-FID (vs real GT) ==="
"$PYTHON" -m wan_ticft.evaluation.compute_real_fid --real_root "$REAL" --work "$OUT/_fid" --limit 24 \
  --candidate "r1s2=$OUT/r1s2/depth_layered_a75/videos" \
  --candidate "r0s1=$OUT/r0s1/depth_layered_a75/videos" \
  --candidate "r1s1=$OUT/r1s1/depth_layered_a75/videos" \
  --candidate "r0s2=$OUT/r0s2/depth_layered_a75/videos"
echo "=== Pareto winner = PSNR/SSIM up AND REAL-FID not worse than r1s2. ==="

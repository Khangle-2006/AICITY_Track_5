#!/bin/bash
set -euo pipefail
# Gate: multi-frame fusion vs a75 single-frame on val BDD24. Pareto = PSNR/SSIM up AND REAL-FID not worse.
cd /workspace
PYTHON=${PYTHON:-/opt/conda/envs/wan22/bin/python}
ROOT=/workspace/experiments/wan_ticft
MANIFEST=$ROOT/ltxv_bdd25_gate/selected_manifest.jsonl
BASE_REPORT=$ROOT/ltxv_bdd25_gate/report.csv
MODEL_CACHE=$ROOT/depth_layered_models/huggingface
REAL=$ROOT/frames_balanced_val_125/val
A75DIR=$ROOT/bdd_supervised_ltx_depth_eval24/depth/depth_layered_a75
OUT=$ROOT/multiframe_gate
MAXH="${MAX_HISTORY:-4}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2}"

echo "=== Multi-frame fusion gate (max_history=$MAXH) ==="
"$PYTHON" -m wan_ticft.evaluation.materialize_depth_layered_multiframe \
  --manifest "$MANIFEST" --base_report "$BASE_REPORT" --out_root "$OUT" \
  --model_cache "$MODEL_CACHE" --view_type bdd --limit 24 --preserve_full_video \
  --alpha 0.75 --max_history "$MAXH" --focal_scale 0.9 --point_stride 2 --splat_radius 1 \
  --min_matches 20 --min_inliers 12 --photo_sigma 32 --segment_batch 4 \
  --pose_smoothing_window 7 --filled_pose_scale 0.35 --confidence_ema 0.65 --clean

MF="$OUT/mf_a75_h${MAXH}"
echo "=== PIXEL (vs GT) ==="
"$PYTHON" -m wan_ticft.evaluation.evaluate_video_reports --manifest "$MANIFEST" \
  --reports a75_single="$A75DIR/report.csv" multiframe="$MF/report.csv" \
  --offset_mode timestamp --out_csv "$OUT/eval_pixel.csv"

echo "=== REAL-FID (vs real GT) ==="
"$PYTHON" -m wan_ticft.evaluation.compute_real_fid --real_root "$REAL" --work "$OUT/_fid" --limit 24 \
  --candidate "a75_single=$A75DIR/videos" --candidate "multiframe=$MF/videos"
echo "=== Pareto: multiframe PSNR/SSIM>=a75 AND REAL-FID<=a75 => build test + submit. ==="

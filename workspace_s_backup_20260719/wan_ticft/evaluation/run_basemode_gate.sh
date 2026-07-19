#!/bin/bash
set -euo pipefail
# Gate: multiframe base_mode video (ghosty cnn_ltxmix) vs repeat_anchor (real frozen, no-ghost) on val BDD24.
# Win = repeat_anchor REAL-FID <= video AND PSNR/SSIM not worse -> fixes ghost cleanly.
cd /workspace
export CPU_THREADS="${CPU_THREADS:-4}"
export OMP_NUM_THREADS=$CPU_THREADS OPENBLAS_NUM_THREADS=$CPU_THREADS MKL_NUM_THREADS=$CPU_THREADS NUMEXPR_NUM_THREADS=$CPU_THREADS VECLIB_MAXIMUM_THREADS=$CPU_THREADS
PYTHON=${PYTHON:-/opt/conda/envs/wan22/bin/python}
ROOT=/workspace/experiments/wan_ticft
MANIFEST=$ROOT/ltxv_bdd25_gate/selected_manifest.jsonl
BASE_REPORT=$ROOT/ltxv_bdd25_gate/report.csv
MODEL_CACHE=$ROOT/depth_layered_models/huggingface
REAL=$ROOT/frames_balanced_val_125/val
A75DIR=$ROOT/bdd_supervised_ltx_depth_eval24/depth/depth_layered_a75
OUT=$ROOT/basemode_gate
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2}"

run() {  # base_mode
  "$PYTHON" -m wan_ticft.evaluation.materialize_depth_layered_multiframe \
    --manifest "$MANIFEST" --base_report "$BASE_REPORT" --out_root "$OUT" \
    --model_cache "$MODEL_CACHE" --view_type bdd --limit 24 --preserve_full_video \
    --alpha 0.75 --max_history 4 --base_mode "$1" --focal_scale 0.9 --point_stride 2 --splat_radius 1 \
    --min_matches 20 --min_inliers 12 --photo_sigma 32 --segment_batch 4 \
    --pose_smoothing_window 7 --filled_pose_scale 0.35 --confidence_ema 0.65
}
echo "=== render video base ==="; run video
echo "=== render repeat_anchor base ==="; run repeat_anchor

V="$OUT/mf_a75_h4_video"; R="$OUT/mf_a75_h4_repeat_anchor"
echo "=== PIXEL vs GT ==="
"$PYTHON" -m wan_ticft.evaluation.evaluate_video_reports --manifest "$MANIFEST" \
  --reports a75_single="$A75DIR/report.csv" mf_video="$V/report.csv" mf_repeat="$R/report.csv" \
  --offset_mode timestamp --out_csv "$OUT/eval_pixel.csv"
echo "=== REAL-FID vs GT ==="
"$PYTHON" -m wan_ticft.evaluation.compute_real_fid --real_root "$REAL" --work "$OUT/_fid" --limit 24 \
  --candidate "a75_single=$A75DIR/videos" --candidate "mf_video=$V/videos" --candidate "mf_repeat=$R/videos"
echo "=== Win = mf_repeat: REAL-FID <= mf_video AND PSNR/SSIM not worse ==="

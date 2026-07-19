#!/usr/bin/env bash
set -euo pipefail

cd /workspace

PYTHON=${PYTHON:-/opt/conda/envs/wan22/bin/python}
ROOT=${ROOT:-/workspace/experiments/wan_ticft}
OUT=${OUT:-$ROOT/temporal_latent_feature_gate24}
MANIFEST=${MANIFEST:-$ROOT/ltxv_bdd25_gate/dense40_manifest.jsonl}
BASE_REPORT=${BASE_REPORT:-$ROOT/cnn_selector_perceptual_tradeoff_gate/cnn_ltxmix65/report.csv}

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-3}"
GPU_ID="${CUDA_VISIBLE_DEVICES%%,*}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

echo "=== Temporal latent feature refiner gate24 ==="
echo "Train/eval split is disabled here: train on the 24-sample BDD gate and score the same 24."
echo "This estimates whether the held-out8 gain scales before spending time on test materialization."
/workspace/wan_ticft/evaluation/check_gpu_free.sh "$GPU_ID" 30000

"$PYTHON" -m wan_ticft.evaluation.train_temporal_latent_refiner \
  --manifest "$MANIFEST" \
  --base_report "$BASE_REPORT" \
  --out_dir "$OUT" \
  --limit 24 \
  --train_rows 24 \
  --epochs 10 \
  --batch_size 1 \
  --width 320 \
  --height 180 \
  --model_width 64 \
  --residual_scale 0.06 \
  --lr 0.0002 \
  --lambda_l1 1.0 \
  --lambda_ssim 0.25 \
  --lambda_residual 0.03 \
  --lambda_temporal 0.10 \
  --lambda_residual_temporal 0.20 \
  --lambda_image_feature 0.04 \
  --lambda_video_feature 0.04 \
  --feature_chunk_size 4 \
  --infer_all \
  --clean

echo
echo "=== Pixel metrics ==="
"$PYTHON" -m wan_ticft.evaluation.evaluate_video_reports \
  --manifest "$MANIFEST" \
  --reports \
    cnn_ltxmix65="$BASE_REPORT" \
    temporal_feature="$OUT/report.csv" \
  --include_repeat_last \
  --offset_mode timestamp \
  --out_csv "${OUT}_eval.csv"

echo
echo "=== Relative FID/FVD proxy ==="
CUDA_VISIBLE_DEVICES="$GPU_ID" \
TORCH_HOME="$ROOT/tools/torch_cache" \
"$PYTHON" -m wan_ticft.evaluation.audit_perceptual_video_distribution \
  --manifest "$MANIFEST" \
  --report cnn_ltxmix65="$BASE_REPORT" \
  --report temporal_feature="$OUT/report.csv" \
  --out_csv "${OUT}_proxy.csv" \
  --clip_frames 16 \
  --frame_stride 4

echo
echo "=== Contact sheets ==="
"$PYTHON" -m baseline.video_contact_sheet \
  --input "$OUT/videos" \
  --mode videos \
  --out_dir "${OUT}_sheets" \
  --samples 12 \
  --tile_width 320 \
  --cols 4

echo
echo "Outputs:"
echo "  model: $OUT/temporal_latent_refiner.pt"
echo "  report: $OUT/report.csv"
echo "  eval: ${OUT}_eval.csv"
echo "  proxy: ${OUT}_proxy.csv"
echo "  sheets: ${OUT}_sheets"
echo
echo "Gate rule:"
echo "  - Continue to test materialization only if temporal_feature improves PSNR by >=0.20,"
echo "    improves or preserves SSIM, and does not worsen FVD proxy versus cnn_ltxmix65."
echo "  - If FID worsens slightly but FVD and pixel improve, allow one test candidate."

#!/usr/bin/env bash
set -euo pipefail

cd /workspace

PYTHON=${PYTHON:-/opt/conda/envs/wan22/bin/python}
ROOT=${ROOT:-/workspace/experiments/wan_ticft}
OUT=${OUT:-$ROOT/temporal_latent_feature_single_gpu_smoke16_heldout8}
MANIFEST=${MANIFEST:-$ROOT/ltxv_bdd25_gate/dense40_manifest.jsonl}
BASE_REPORT=${BASE_REPORT:-$ROOT/cnn_selector_perceptual_tradeoff_gate/cnn_ltxmix65/report.csv}

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-3}"
GPU_ID="${CUDA_VISIBLE_DEVICES%%,*}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

echo "=== Temporal latent refiner feature correction: single-GPU batch 1 ==="
echo "Keeps Inception/R3D feature losses, reduces batch size to test OOM."
/workspace/wan_ticft/evaluation/check_gpu_free.sh "$GPU_ID" 30000

"$PYTHON" -m wan_ticft.evaluation.train_temporal_latent_refiner \
  --manifest "$MANIFEST" \
  --base_report "$BASE_REPORT" \
  --out_dir "$OUT" \
  --limit 24 \
  --train_rows 16 \
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
  --clean

echo
echo "=== Pixel metrics ==="
"$PYTHON" -m wan_ticft.evaluation.evaluate_video_reports \
  --manifest "$OUT/eval_manifest.jsonl" \
  --reports \
    cnn_ltxmix65="$BASE_REPORT" \
    temporal_feature_single="$OUT/report.csv" \
  --include_repeat_last \
  --offset_mode timestamp \
  --out_csv "${OUT}_eval.csv"

echo
echo "=== Relative FID/FVD proxy ==="
CUDA_VISIBLE_DEVICES="$GPU_ID" \
TORCH_HOME="$ROOT/tools/torch_cache" \
"$PYTHON" -m wan_ticft.evaluation.audit_perceptual_video_distribution \
  --manifest "$OUT/eval_manifest.jsonl" \
  --report cnn_ltxmix65="$BASE_REPORT" \
  --report temporal_feature_single="$OUT/report.csv" \
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
echo "  eval: ${OUT}_eval.csv"
echo "  proxy: ${OUT}_proxy.csv"
echo "  sheets: ${OUT}_sheets"

#!/usr/bin/env bash
set -euo pipefail

cd /workspace

PYTHON=${PYTHON:-/opt/conda/envs/wan22/bin/python}
ROOT=${ROOT:-/workspace/experiments/wan_ticft}
TRAIN_MANIFEST=${TRAIN_MANIFEST:-$ROOT/manifests/train_ticft_balanced_300_frames.jsonl}
EVAL_MANIFEST=${EVAL_MANIFEST:-$ROOT/ltxv_bdd25_gate/dense40_manifest.jsonl}
TRAIN_BASE=${TRAIN_BASE:-$ROOT/bdd_train100_region_motion_base}
EVAL_BASE_REPORT=${EVAL_BASE_REPORT:-$ROOT/cnn_selector_perceptual_tradeoff_gate/cnn_ltxmix65/report.csv}
OUT=${OUT:-$ROOT/temporal_latent_feature_train100_gate24}

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-3}"
GPU_ID="${CUDA_VISIBLE_DEVICES%%,*}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

echo "=== Temporal latent feature refiner: train100 -> BDD24 validation ==="
echo "Train base: region-motion BDD train samples"
echo "Eval base: cnn_ltxmix65 BDD24"
/workspace/wan_ticft/evaluation/check_gpu_free.sh "$GPU_ID" 30000

echo
echo "[1/4] Materialize BDD train100 region-motion base"
"$PYTHON" -m wan_ticft.evaluation.materialize_region_motion_report \
  --manifest "$TRAIN_MANIFEST" \
  --out_dir "$TRAIN_BASE" \
  --limit 100

echo
echo "[2/4] Train on train100, infer on BDD24"
"$PYTHON" -m wan_ticft.evaluation.train_temporal_latent_refiner \
  --manifest "$TRAIN_MANIFEST" \
  --base_report "$TRAIN_BASE/report.csv" \
  --train_manifest "$TRAIN_BASE/selected_manifest.jsonl" \
  --train_base_report "$TRAIN_BASE/report.csv" \
  --eval_manifest "$EVAL_MANIFEST" \
  --eval_base_report "$EVAL_BASE_REPORT" \
  --out_dir "$OUT" \
  --limit 24 \
  --train_limit 100 \
  --eval_limit 24 \
  --train_rows 100 \
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
echo "[3/4] Pixel metrics"
"$PYTHON" -m wan_ticft.evaluation.evaluate_video_reports \
  --manifest "$EVAL_MANIFEST" \
  --reports \
    cnn_ltxmix65="$EVAL_BASE_REPORT" \
    temporal_train100="$OUT/report.csv" \
  --include_repeat_last \
  --offset_mode timestamp \
  --out_csv "${OUT}_eval.csv"

echo
echo "[4/4] Relative FID/FVD proxy"
CUDA_VISIBLE_DEVICES="$GPU_ID" \
TORCH_HOME="$ROOT/tools/torch_cache" \
"$PYTHON" -m wan_ticft.evaluation.audit_perceptual_video_distribution \
  --manifest "$EVAL_MANIFEST" \
  --report cnn_ltxmix65="$EVAL_BASE_REPORT" \
  --report temporal_train100="$OUT/report.csv" \
  --out_csv "${OUT}_proxy.csv" \
  --clip_frames 16 \
  --frame_stride 4

"$PYTHON" -m baseline.video_contact_sheet \
  --input "$OUT/videos" \
  --mode videos \
  --out_dir "${OUT}_sheets" \
  --samples 12 \
  --tile_width 320 \
  --cols 4

echo
echo "Outputs:"
echo "  train_base: $TRAIN_BASE"
echo "  model: $OUT/temporal_latent_refiner.pt"
echo "  eval: ${OUT}_eval.csv"
echo "  proxy: ${OUT}_proxy.csv"
echo "  sheets: ${OUT}_sheets"
echo
echo "Gate rule:"
echo "  - Continue only if train100 refiner improves BDD24 PSNR by >=0.15,"
echo "    preserves SSIM, and does not worsen FVD proxy versus cnn_ltxmix65."
echo "  - If it fails, do not train larger; the train-base mismatch is too high."

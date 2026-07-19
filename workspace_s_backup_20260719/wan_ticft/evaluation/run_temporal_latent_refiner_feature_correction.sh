#!/usr/bin/env bash
set -euo pipefail

echo "=== Temporal latent refiner feature-regularized correction ==="
echo "Keeps sequence-level refiner, adds Inception/R3D feature losses to reduce proxy drift."

cd /workspace
PYTHON=${PYTHON:-/opt/conda/envs/wan22/bin/python}
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-3}
export CUDA_VISIBLE_DEVICES
export PYTHONPATH=/workspace:/workspace/wan_ticft:${PYTHONPATH:-}

MANIFEST=${MANIFEST:-/workspace/experiments/wan_ticft/ltxv_bdd25_gate/dense40_manifest.jsonl}
BASE_REPORT=${BASE_REPORT:-/workspace/experiments/wan_ticft/cnn_selector_perceptual_tradeoff_gate/cnn_ltxmix65/report.csv}
OUT=${OUT:-/workspace/experiments/wan_ticft/temporal_latent_feature_smoke16_heldout8}

"$PYTHON" -m wan_ticft.evaluation.train_temporal_latent_refiner \
  --manifest "$MANIFEST" \
  --base_report "$BASE_REPORT" \
  --out_dir "$OUT" \
  --limit 24 \
  --train_rows 16 \
  --width 256 \
  --height 144 \
  --model_width 48 \
  --residual_scale 0.06 \
  --epochs 8 \
  --batch_size 1 \
  --lr 2e-4 \
  --lambda_l1 0.35 \
  --lambda_ssim 0.16 \
  --lambda_temporal 0.22 \
  --lambda_residual 0.16 \
  --lambda_residual_temporal 0.30 \
  --lambda_image_feature 0.00 \
  --lambda_video_feature 0.02 \
  --clean

"$PYTHON" -m wan_ticft.evaluation.evaluate_video_reports \
  --manifest "$OUT/eval_manifest.jsonl" \
  --reports \
    cnn_ltxmix65="$BASE_REPORT" \
    temporal_feature="$OUT/report.csv" \
  --include_repeat_last \
  --offset_mode timestamp \
  --out_csv "${OUT}_eval.csv"

"$PYTHON" -m wan_ticft.evaluation.audit_perceptual_video_distribution \
  --manifest "$OUT/eval_manifest.jsonl" \
  --report cnn_ltxmix65="$BASE_REPORT" \
  --report temporal_feature="$OUT/report.csv" \
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

cat <<EOF

Outputs:
  model: $OUT/temporal_latent_refiner.pt
  eval: ${OUT}_eval.csv
  proxy: ${OUT}_proxy.csv
  sheets: ${OUT}_sheets

Gate rule:
  - Continue temporal latent only if this keeps most of the PSNR gain and brings
    FID/FVD proxy back to cnn_ltxmix65 or better.
  - If proxy remains worse, keep temporal latent only as a possible pixel-score
    submission branch, not as the FID/FVD breakthrough path.
EOF

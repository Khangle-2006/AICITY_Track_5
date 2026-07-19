#!/usr/bin/env bash
set -euo pipefail

echo "=== Temporal latent refiner MAX feature correction ==="
echo "Keeps 320x180, model_width=64, Inception + R3D losses, chunked features, DataParallel."

cd /workspace
PYTHON=${PYTHON:-/opt/conda/envs/wan22/bin/python}
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-2,3}
export CUDA_VISIBLE_DEVICES
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
export PYTHONPATH=/workspace:/workspace/wan_ticft:${PYTHONPATH:-}

MANIFEST=${MANIFEST:-/workspace/experiments/wan_ticft/ltxv_bdd25_gate/dense40_manifest.jsonl}
BASE_REPORT=${BASE_REPORT:-/workspace/experiments/wan_ticft/cnn_selector_perceptual_tradeoff_gate/cnn_ltxmix65/report.csv}
OUT=${OUT:-/workspace/experiments/wan_ticft/temporal_latent_feature_max_smoke16_heldout8}

"$PYTHON" -m wan_ticft.evaluation.train_temporal_latent_refiner \
  --manifest "$MANIFEST" \
  --base_report "$BASE_REPORT" \
  --out_dir "$OUT" \
  --limit 24 \
  --train_rows 16 \
  --width 320 \
  --height 180 \
  --model_width 64 \
  --residual_scale 0.06 \
  --epochs 10 \
  --batch_size 2 \
  --lr 2e-4 \
  --lambda_l1 0.35 \
  --lambda_ssim 0.16 \
  --lambda_temporal 0.22 \
  --lambda_residual 0.16 \
  --lambda_residual_temporal 0.30 \
  --lambda_image_feature 0.02 \
  --lambda_video_feature 0.02 \
  --feature_chunk_size 8 \
  --data_parallel \
  --clean

"$PYTHON" -m wan_ticft.evaluation.evaluate_video_reports \
  --manifest "$OUT/eval_manifest.jsonl" \
  --reports \
    cnn_ltxmix65="$BASE_REPORT" \
    temporal_feature_max="$OUT/report.csv" \
  --include_repeat_last \
  --offset_mode timestamp \
  --out_csv "${OUT}_eval.csv"

"$PYTHON" -m wan_ticft.evaluation.audit_perceptual_video_distribution \
  --manifest "$OUT/eval_manifest.jsonl" \
  --report cnn_ltxmix65="$BASE_REPORT" \
  --report temporal_feature_max="$OUT/report.csv" \
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
  - This is the main temporal-latent correction. Continue only if it keeps most
    of the PSNR/SSIM gain and improves or nearly matches cnn_ltxmix65 FID/FVD.
  - If this still worsens proxy, temporal latent is pixel-positive but not a
    FID/FVD breakthrough.
EOF

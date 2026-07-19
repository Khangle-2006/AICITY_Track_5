#!/usr/bin/env bash
set -euo pipefail

echo "=== Temporal latent refiner smoke ==="
echo "Sequence-level 3DConv low-res latent residual over cnn_ltxmix65."
echo "This tests motion/temporal consistency, not frame-wise RGB cleanup."

cd /workspace
PYTHON=${PYTHON:-/opt/conda/envs/wan22/bin/python}
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-3}
export CUDA_VISIBLE_DEVICES
export PYTHONPATH=/workspace:/workspace/wan_ticft:${PYTHONPATH:-}

MANIFEST=${MANIFEST:-/workspace/experiments/wan_ticft/ltxv_bdd25_gate/dense40_manifest.jsonl}
BASE_REPORT=${BASE_REPORT:-/workspace/experiments/wan_ticft/cnn_selector_perceptual_tradeoff_gate/cnn_ltxmix65/report.csv}
OUT=${OUT:-/workspace/experiments/wan_ticft/temporal_latent_refiner_smoke16_heldout8}

"$PYTHON" -m wan_ticft.evaluation.train_temporal_latent_refiner \
  --manifest "$MANIFEST" \
  --base_report "$BASE_REPORT" \
  --out_dir "$OUT" \
  --limit 24 \
  --train_rows 16 \
  --width 320 \
  --height 180 \
  --model_width 64 \
  --residual_scale 0.08 \
  --epochs 12 \
  --batch_size 2 \
  --lr 2e-4 \
  --lambda_l1 0.45 \
  --lambda_ssim 0.20 \
  --lambda_temporal 0.20 \
  --lambda_residual 0.12 \
  --lambda_residual_temporal 0.25 \
  --clean

"$PYTHON" -m wan_ticft.evaluation.evaluate_video_reports \
  --manifest "$OUT/eval_manifest.jsonl" \
  --reports \
    cnn_ltxmix65="$BASE_REPORT" \
    temporal_latent="$OUT/report.csv" \
  --include_repeat_last \
  --offset_mode timestamp \
  --out_csv "${OUT}_eval.csv"

"$PYTHON" -m wan_ticft.evaluation.audit_perceptual_video_distribution \
  --manifest "$OUT/eval_manifest.jsonl" \
  --report cnn_ltxmix65="$BASE_REPORT" \
  --report temporal_latent="$OUT/report.csv" \
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
  - Continue only if temporal_latent improves FID/FVD proxy or clearly reduces
    temporal artifacts while keeping PSNR/SSIM near cnn_ltxmix65.
  - If it behaves like RGB adapters and worsens both proxy metrics, close this
    learned-refiner family and move to explicit motion/geometry conditioning.
EOF

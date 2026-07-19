#!/usr/bin/env bash
set -euo pipefail

echo "=== Adversarial residual adapter smoke ==="
echo "Train on 16 BDD val samples, evaluate on 8 held-out samples."
echo "Uses image/video discriminators to directly pressure target-like distribution."

cd /workspace
PYTHON=${PYTHON:-/opt/conda/envs/wan22/bin/python}
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-3}
export CUDA_VISIBLE_DEVICES
export PYTHONPATH=/workspace:/workspace/wan_ticft:${PYTHONPATH:-}

MANIFEST=${MANIFEST:-/workspace/experiments/wan_ticft/ltxv_bdd25_gate/dense40_manifest.jsonl}
BASE_REPORT=${BASE_REPORT:-/workspace/experiments/wan_ticft/cnn_selector_perceptual_tradeoff_gate/cnn_ltxmix65/report.csv}
OUT=${OUT:-/workspace/experiments/wan_ticft/metric_adversarial_adapter_smoke16_heldout8}

"$PYTHON" -m wan_ticft.evaluation.train_adversarial_residual_adapter \
  --manifest "$MANIFEST" \
  --base_report "$BASE_REPORT" \
  --out_dir "$OUT" \
  --limit 24 \
  --train_rows 16 \
  --width 320 \
  --height 180 \
  --model_width 48 \
  --residual_scale 0.06 \
  --epochs 4 \
  --clip_frames 8 \
  --sequence_batch 2 \
  --max_steps 48 \
  --lr 2e-4 \
  --discriminator_lr 1e-4 \
  --lambda_l1 0.20 \
  --lambda_ssim 0.08 \
  --lambda_temporal 0.08 \
  --lambda_residual 0.18 \
  --lambda_image_gan 0.04 \
  --lambda_video_gan 0.04 \
  --clean

"$PYTHON" -m wan_ticft.evaluation.evaluate_video_reports \
  --manifest "$OUT/eval_manifest.jsonl" \
  --reports \
    cnn_ltxmix65="$BASE_REPORT" \
    adversarial_adapter="$OUT/report.csv" \
  --include_repeat_last \
  --offset_mode timestamp \
  --out_csv "${OUT}_eval.csv"

"$PYTHON" -m wan_ticft.evaluation.audit_perceptual_video_distribution \
  --manifest "$OUT/eval_manifest.jsonl" \
  --report cnn_ltxmix65="$BASE_REPORT" \
  --report adversarial_adapter="$OUT/report.csv" \
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
  adapter: $OUT/adapter.pt
  eval: ${OUT}_eval.csv
  proxy: ${OUT}_proxy.csv
  sheets: ${OUT}_sheets

Gate rule:
  - Continue only if adversarial_adapter keeps a PSNR/SSIM gain and improves
    FID or FVD proxy without materially worsening the other.
  - If it only improves pixel and still worsens both FID/FVD, close adapter as
    a leaderboard direction despite the PSNR gain.
EOF

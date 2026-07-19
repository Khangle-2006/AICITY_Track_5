#!/usr/bin/env bash
set -euo pipefail

echo "=== Metric-loss residual adapter smoke ==="
echo "Train on 16 BDD val samples, evaluate on 8 held-out samples."

cd /workspace
PYTHON=${PYTHON:-/opt/conda/envs/wan22/bin/python}
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export CUDA_VISIBLE_DEVICES
export PYTHONPATH=/workspace/wan_ticft:${PYTHONPATH:-}

MANIFEST=${MANIFEST:-/workspace/experiments/wan_ticft/ltxv_bdd25_gate/dense40_manifest.jsonl}
BASE_REPORT=${BASE_REPORT:-/workspace/experiments/wan_ticft/cnn_selector_perceptual_tradeoff_gate/cnn_ltxmix65/report.csv}
OUT=${OUT:-/workspace/experiments/wan_ticft/metric_residual_adapter_smoke16_heldout8}

"$PYTHON" -m wan_ticft.evaluation.train_metric_residual_adapter \
  --manifest "$MANIFEST" \
  --base_report "$BASE_REPORT" \
  --out_dir "$OUT" \
  --limit 24 \
  --train_rows 16 \
  --width 320 \
  --height 180 \
  --model_width 48 \
  --residual_scale 0.08 \
  --epochs 5 \
  --batch_size 16 \
  --lr 3e-4 \
  --lambda_ssim 0.35 \
  --lambda_grad 0.08 \
  --lambda_residual 0.10 \
  --clean

"$PYTHON" -m wan_ticft.evaluation.evaluate_video_reports \
  --manifest "$OUT/eval_manifest.jsonl" \
  --reports \
    cnn_ltxmix65="$BASE_REPORT" \
    metric_adapter="$OUT/report.csv" \
  --include_repeat_last \
  --offset_mode timestamp \
  --out_csv "${OUT}_eval.csv"

"$PYTHON" -m wan_ticft.evaluation.audit_perceptual_video_distribution \
  --manifest "$OUT/eval_manifest.jsonl" \
  --report cnn_ltxmix65="$BASE_REPORT" \
  --report metric_adapter="$OUT/report.csv" \
  --out_csv "${OUT}_proxy.csv" \
  --clip_frames 16 \
  --frame_stride 4

cat <<EOF

Outputs:
  adapter: $OUT/adapter.pt
  eval: ${OUT}_eval.csv
  proxy: ${OUT}_proxy.csv

Gate rule:
  - Continue only if metric_adapter improves held-out FID and FVD proxy
    without losing more than 0.05 PSNR / 0.002 SSIM versus cnn_ltxmix65.
  - If it fails, close metric-loss output adapter.
EOF

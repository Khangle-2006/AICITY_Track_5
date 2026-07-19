#!/usr/bin/env bash
set -euo pipefail

echo "=== Metric-adapter/base mix tradeoff gate ==="
echo "Uses the pixel-strong adapter v1 and searches for a FID/FVD-safe residual strength."

cd /workspace
PYTHON=${PYTHON:-/opt/conda/envs/wan22/bin/python}
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-3}
export CUDA_VISIBLE_DEVICES
export PYTHONPATH=/workspace:/workspace/wan_ticft:${PYTHONPATH:-}

BASE_REPORT=${BASE_REPORT:-/workspace/experiments/wan_ticft/cnn_selector_perceptual_tradeoff_gate/cnn_ltxmix65/report.csv}
ADAPTER_DIR=${ADAPTER_DIR:-/workspace/experiments/wan_ticft/metric_residual_adapter_smoke16_heldout8}
ADAPTER_REPORT=${ADAPTER_REPORT:-$ADAPTER_DIR/report.csv}
MANIFEST=${MANIFEST:-$ADAPTER_DIR/eval_manifest.jsonl}
OUT=${OUT:-/workspace/experiments/wan_ticft/metric_adapter_mix_tradeoff_heldout8}

"$PYTHON" -m wan_ticft.evaluation.materialize_video_report_mix \
  --report_a "$BASE_REPORT" \
  --report_b "$ADAPTER_REPORT" \
  --out_root "$OUT" \
  --variant adapter_mix25=0.25 \
  --variant adapter_mix40=0.40 \
  --variant adapter_mix55=0.55 \
  --variant adapter_mix70=0.70 \
  --clean

"$PYTHON" -m wan_ticft.evaluation.evaluate_video_reports \
  --manifest "$MANIFEST" \
  --reports \
    cnn_ltxmix65="$BASE_REPORT" \
    metric_adapter="$ADAPTER_REPORT" \
    adapter_mix25="$OUT/adapter_mix25/report.csv" \
    adapter_mix40="$OUT/adapter_mix40/report.csv" \
    adapter_mix55="$OUT/adapter_mix55/report.csv" \
    adapter_mix70="$OUT/adapter_mix70/report.csv" \
  --include_repeat_last \
  --offset_mode timestamp \
  --out_csv "${OUT}_eval.csv"

"$PYTHON" -m wan_ticft.evaluation.audit_perceptual_video_distribution \
  --manifest "$MANIFEST" \
  --report cnn_ltxmix65="$BASE_REPORT" \
  --report metric_adapter="$ADAPTER_REPORT" \
  --report adapter_mix25="$OUT/adapter_mix25/report.csv" \
  --report adapter_mix40="$OUT/adapter_mix40/report.csv" \
  --report adapter_mix55="$OUT/adapter_mix55/report.csv" \
  --report adapter_mix70="$OUT/adapter_mix70/report.csv" \
  --out_csv "${OUT}_proxy.csv" \
  --clip_frames 16 \
  --frame_stride 4

for label in adapter_mix25 adapter_mix40 adapter_mix55 adapter_mix70; do
  "$PYTHON" -m baseline.video_contact_sheet \
    --input "$OUT/$label/videos" \
    --mode videos \
    --out_dir "${OUT}_${label}_sheets" \
    --samples 12 \
    --tile_width 320 \
    --cols 4
done

cat <<EOF

Outputs:
  eval: ${OUT}_eval.csv
  proxy: ${OUT}_proxy.csv
  videos: $OUT

Gate rule:
  - Continue adapter direction only if a mix keeps at least +0.15 PSNR over
    cnn_ltxmix65 and reduces both FID/FVD proxy versus raw metric_adapter.
  - Promote only if FID/FVD are near cnn_ltxmix65 or better; otherwise adapter
    remains pixel-only and should not be used as the main submission path.
EOF

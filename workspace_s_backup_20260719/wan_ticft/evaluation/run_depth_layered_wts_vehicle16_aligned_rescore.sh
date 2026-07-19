#!/usr/bin/env bash
set -euo pipefail

cd /workspace

PYTHON=/opt/conda/envs/wan22/bin/python
ROOT=/workspace/experiments/wan_ticft
SOURCE="$ROOT/depth_layered_wts_vehicle16"
OUT="$ROOT/depth_layered_wts_vehicle16_aligned"
MANIFEST="$SOURCE/selected_manifest.jsonl"
GPU_ID="${CUDA_VISIBLE_DEVICES:-3}"
GPU_ID="${GPU_ID%%,*}"

echo "=== Correctly aligned WTS vehicle16 rescore (no inference) ==="
"$PYTHON" -m wan_ticft.evaluation.prepend_video_guard_reports \
  --report lossless_control="$SOURCE/depth_layered_a00/report.csv" \
  --report depth_a75="$SOURCE/depth_layered_a75/report.csv" \
  --report depth_hdecay="$SOURCE/depth_layered_hdecay/report.csv" \
  --out_root "$OUT"

"$PYTHON" -m wan_ticft.evaluation.evaluate_video_reports \
  --manifest "$MANIFEST" \
  --reports \
    lossless_control="$OUT/lossless_control/report.csv" \
    depth_a75="$OUT/depth_a75/report.csv" \
    depth_hdecay="$OUT/depth_hdecay/report.csv" \
  --include_repeat_last \
  --offset_mode timestamp \
  --out_csv "${OUT}_eval.csv"

CUDA_VISIBLE_DEVICES="$GPU_ID" TORCH_HOME="$ROOT/tools/torch_cache" \
"$PYTHON" -m wan_ticft.evaluation.audit_perceptual_video_distribution \
  --manifest "$MANIFEST" \
  --report lossless_control="$OUT/lossless_control/report.csv" \
  --report depth_a75="$OUT/depth_a75/report.csv" \
  --report depth_hdecay="$OUT/depth_hdecay/report.csv" \
  --out_csv "${OUT}_proxy.csv" \
  --clip_frames 8 \
  --frame_stride 1

echo
echo "eval=${OUT}_eval.csv"
echo "proxy=${OUT}_proxy.csv"
echo "This run only prepends guard frames and rescores existing outputs."

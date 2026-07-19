#!/usr/bin/env bash
set -euo pipefail

cd /workspace

PYTHON=/opt/conda/envs/wan22/bin/python
ROOT=/workspace/experiments/wan_ticft
MANIFEST="$ROOT/ltxv_bdd25_gate/dense40_manifest.jsonl"
RAW_REPORT="$ROOT/ltxv_bdd25_gate/report.csv"
OUT_ROOT="$ROOT/ltxv_bdd25_transition_duration_gate"
GPU_ID="${CUDA_VISIBLE_DEVICES:-3}"
GPU_ID="${GPU_ID%%,*}"

echo "=== Bounded LTX transition-duration perceptual gate ==="
echo "Testing only transition24 and transition32 against promoted transition16."

"$PYTHON" -m wan_ticft.evaluation.build_ltx_condition_transitions \
  --manifest "$MANIFEST" \
  --ltx_report "$RAW_REPORT" \
  --out_root "$OUT_ROOT" \
  --transitions 24 32 \
  --clean

echo
echo "=== Pixel metrics ==="
"$PYTHON" -m wan_ticft.evaluation.evaluate_video_reports \
  --manifest "$MANIFEST" \
  --reports \
    "transition16=$ROOT/ltxv_bdd25_transitions/ltx_transition_16/report.csv" \
    "transition24=$OUT_ROOT/ltx_transition_24/report.csv" \
    "transition32=$OUT_ROOT/ltx_transition_32/report.csv" \
  --out_csv "$ROOT/ltx_transition_duration_gate_eval.csv" \
  --offset_mode timestamp

echo
echo "=== Relative FID/FVD proxy ==="
/workspace/wan_ticft/evaluation/check_gpu_free.sh "$GPU_ID" 3000
CUDA_VISIBLE_DEVICES="$GPU_ID" \
TORCH_HOME="$ROOT/tools/torch_cache" \
"$PYTHON" -m wan_ticft.evaluation.audit_perceptual_video_distribution \
  --manifest "$MANIFEST" \
  --report "transition16=$ROOT/ltxv_bdd25_transitions/ltx_transition_16/report.csv" \
  --report "transition24=$OUT_ROOT/ltx_transition_24/report.csv" \
  --report "transition32=$OUT_ROOT/ltx_transition_32/report.csv" \
  --out_csv "$ROOT/ltx_transition_duration_perceptual_proxy.csv"

echo
echo "Final transition-duration rule:"
echo "  - Promote only if transition24 or transition32 improves both FID and FVD"
echo "    proxies versus transition16, loses no more than 0.15 PSNR and 0.003 SSIM,"
echo "    and passes visual QA."
echo "  - If neither passes, retain transition16 and close transition-duration tuning."

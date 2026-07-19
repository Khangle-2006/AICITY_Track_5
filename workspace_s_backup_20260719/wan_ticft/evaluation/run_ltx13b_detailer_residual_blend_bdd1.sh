#!/usr/bin/env bash
set -euo pipefail

cd /workspace

PYTHON=/opt/conda/envs/wan22/bin/python
ROOT=/workspace/experiments/wan_ticft
SOURCE="$ROOT/ltx13b_ic_lora_detailer_bdd1_smoke"
BASE_REPORT="$ROOT/sa2va_actor_depth_strength_extension_bdd24/actor_depth_a125/report.csv"
DETAILER_REPORT="$SOURCE/detailer_s100/report.csv"
OUT="$ROOT/ltx13b_detailer_residual_blend_bdd1"
SAMPLE=val_bdd_pc_5k_video7040_phase0
GPU_ID="${CUDA_VISIBLE_DEVICES:-0}"
GPU_ID="${GPU_ID%%,*}"

echo "=== LTX 13B detailer color-matched residual blend BDD1 ==="
for path in "$SOURCE/selected_manifest.jsonl" "$BASE_REPORT" "$DETAILER_REPORT"; do
  [[ -s "$path" ]] || { echo "Missing input: $path" >&2; exit 2; }
done
rm -rf "$OUT"

"$PYTHON" -m wan_ticft.evaluation.materialize_detailer_residual_blends \
  --base_report "$BASE_REPORT" \
  --detailer_report "$DETAILER_REPORT" \
  --out_dir "$OUT" \
  --sample_id "$SAMPLE" \
  --alpha 0.10 \
  --alpha 0.20 \
  --alpha 0.30

"$PYTHON" -m wan_ticft.evaluation.evaluate_video_reports \
  --manifest "$SOURCE/selected_manifest.jsonl" \
  --reports \
    actor_depth_a125="$BASE_REPORT" \
    detailer_s100="$DETAILER_REPORT" \
    cmblend_a10="$OUT/cmblend_a10/report.csv" \
    cmblend_a20="$OUT/cmblend_a20/report.csv" \
    cmblend_a30="$OUT/cmblend_a30/report.csv" \
  --include_repeat_last \
  --offset_mode timestamp \
  --out_csv "${OUT}_eval.csv"

/workspace/wan_ticft/evaluation/check_gpu_free.sh "$GPU_ID" 7000
CUDA_VISIBLE_DEVICES="$GPU_ID" TORCH_HOME="$ROOT/tools/torch_cache" \
"$PYTHON" -m wan_ticft.evaluation.audit_perceptual_video_distribution \
  --manifest "$SOURCE/selected_manifest.jsonl" \
  --report actor_depth_a125="$BASE_REPORT" \
  --report detailer_s100="$DETAILER_REPORT" \
  --report cmblend_a10="$OUT/cmblend_a10/report.csv" \
  --report cmblend_a20="$OUT/cmblend_a20/report.csv" \
  --report cmblend_a30="$OUT/cmblend_a30/report.csv" \
  --out_csv "${OUT}_proxy.csv" \
  --clip_frames 16 \
  --frame_stride 4

"$PYTHON" -m wan_ticft.evaluation.make_report_comparison_sheet \
  --manifest "$SOURCE/selected_manifest.jsonl" \
  --reports \
    actor_depth_a125="$BASE_REPORT" \
    detailer_s100="$DETAILER_REPORT" \
    cmblend_a10="$OUT/cmblend_a10/report.csv" \
    cmblend_a20="$OUT/cmblend_a20/report.csv" \
    cmblend_a30="$OUT/cmblend_a30/report.csv" \
  --sample_id "$SAMPLE" \
  --out_dir "${OUT}_compare_sheets" \
  --frames 8 \
  --tile_width 260

echo
echo "eval=${OUT}_eval.csv"
echo "proxy=${OUT}_proxy.csv"
echo "sheets=${OUT}_compare_sheets"
echo "Continue only if a blend improves both FID and FVD while losing <=0.10 dB PSNR and <=0.002 SSIM."

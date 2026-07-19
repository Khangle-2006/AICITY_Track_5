#!/usr/bin/env bash
set -euo pipefail

cd /workspace

PYTHON=/opt/conda/envs/wan22/bin/python
ROOT=/workspace/experiments/wan_ticft
SOURCE="$ROOT/ltx13b_ic_lora_detailer_bdd4_gate"
BASE_REPORT="$ROOT/sa2va_actor_depth_strength_extension_bdd24/actor_depth_a125/report.csv"
DETAILER_REPORT="$SOURCE/detailer_s100/report.csv"
OUT="$ROOT/ltx13b_detailer_lowblend_bdd4_gate"
GPU_ID="${CUDA_VISIBLE_DEVICES:-0}"
GPU_ID="${GPU_ID%%,*}"
SAMPLES=(
  val_bdd_pc_5k_video7040_phase0
  val_bdd_pc_5k_video8125_phase1
  val_bdd_pc_5k_video8139_phase2
  val_bdd_pc_5k_video9450_phase4
)

echo "=== LTX 13B detailer low-strength residual sweep BDD4 ==="
echo "Reuses existing s100 detailer videos; no LTX inference is run."
for path in "$SOURCE/selected_manifest.jsonl" "$BASE_REPORT" "$DETAILER_REPORT"; do
  [[ -s "$path" ]] || { echo "Missing input: $path" >&2; exit 2; }
done

SAMPLE_ARGS=()
for sample_id in "${SAMPLES[@]}"; do
  SAMPLE_ARGS+=(--sample_id "$sample_id")
done

rm -rf "$OUT" "${OUT}_compare_sheets"

echo "[1/4] Materialize color-matched residual blends"
"$PYTHON" -m wan_ticft.evaluation.materialize_detailer_residual_blends \
  --base_report "$BASE_REPORT" \
  --detailer_report "$DETAILER_REPORT" \
  --out_dir "$OUT" \
  "${SAMPLE_ARGS[@]}" \
  --alpha 0.03 \
  --alpha 0.05 \
  --alpha 0.07

echo "[2/4] Pixel metrics"
"$PYTHON" -m wan_ticft.evaluation.evaluate_video_reports \
  --manifest "$SOURCE/selected_manifest.jsonl" \
  --reports \
    actor_depth_a125="$BASE_REPORT" \
    cmblend_a03="$OUT/cmblend_a03/report.csv" \
    cmblend_a05="$OUT/cmblend_a05/report.csv" \
    cmblend_a07="$OUT/cmblend_a07/report.csv" \
    cmblend_a10="$ROOT/ltx13b_detailer_cmblend_a10_bdd4_gate/cmblend_a10/report.csv" \
  --include_repeat_last \
  --offset_mode timestamp \
  --out_csv "${OUT}_eval.csv"

echo "[3/4] FID/FVD proxy"
/workspace/wan_ticft/evaluation/check_gpu_free.sh "$GPU_ID" 7000
CUDA_VISIBLE_DEVICES="$GPU_ID" TORCH_HOME="$ROOT/tools/torch_cache" \
"$PYTHON" -m wan_ticft.evaluation.audit_perceptual_video_distribution \
  --manifest "$SOURCE/selected_manifest.jsonl" \
  --report actor_depth_a125="$BASE_REPORT" \
  --report cmblend_a03="$OUT/cmblend_a03/report.csv" \
  --report cmblend_a05="$OUT/cmblend_a05/report.csv" \
  --report cmblend_a07="$OUT/cmblend_a07/report.csv" \
  --report cmblend_a10="$ROOT/ltx13b_detailer_cmblend_a10_bdd4_gate/cmblend_a10/report.csv" \
  --out_csv "${OUT}_proxy.csv" \
  --clip_frames 16 \
  --frame_stride 4

echo "[4/4] Contact sheets"
for sample_id in "${SAMPLES[@]}"; do
  "$PYTHON" -m wan_ticft.evaluation.make_report_comparison_sheet \
    --manifest "$SOURCE/selected_manifest.jsonl" \
    --reports \
      actor_depth_a125="$BASE_REPORT" \
      cmblend_a03="$OUT/cmblend_a03/report.csv" \
      cmblend_a05="$OUT/cmblend_a05/report.csv" \
      cmblend_a07="$OUT/cmblend_a07/report.csv" \
      cmblend_a10="$ROOT/ltx13b_detailer_cmblend_a10_bdd4_gate/cmblend_a10/report.csv" \
    --sample_id "$sample_id" \
    --out_dir "${OUT}_compare_sheets" \
    --frames 8 \
    --tile_width 260
done

echo
echo "eval=${OUT}_eval.csv"
echo "proxy=${OUT}_proxy.csv"
echo "sheets=${OUT}_compare_sheets"
echo "Promote only if one blend improves both FID and FVD versus actor_depth_a125,"
echo "loses no more than 0.05 dB PSNR / 0.001 SSIM, and remains visually clean."

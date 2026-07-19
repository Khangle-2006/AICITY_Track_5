#!/usr/bin/env bash
set -euo pipefail

cd /workspace

TRAIN_PYTHON=/workspace/experiments/wan_ticft/ltxv_trainer_env/bin/python
PYTHON=/opt/conda/envs/wan22/bin/python
ROOT=/workspace/experiments/wan_ticft
REPO=/workspace/third_party/LTX-Video-Trainer
MODEL_DIR=/workspace/models/ltx13b098_ic_detailer
MODEL="$MODEL_DIR/ltxv-13b-0.9.8-dev.safetensors"
LORA="$MODEL_DIR/ltxv-098-ic-lora-detailer-diffusers.safetensors"
MANIFEST="$ROOT/ltxv_bdd25_gate/dense40_manifest.jsonl"
BASE_REPORT="$ROOT/sa2va_actor_depth_strength_extension_bdd24/actor_depth_a125/report.csv"
BDD1="$ROOT/ltx13b_ic_lora_detailer_bdd1_smoke"
OUT="$ROOT/ltx13b_ic_lora_detailer_bdd4_gate"
BLEND="$ROOT/ltx13b_detailer_cmblend_a10_bdd4_gate"
GPU_ID="${CUDA_VISIBLE_DEVICES:-0}"
GPU_ID="${GPU_ID%%,*}"
SAMPLES=(
  val_bdd_pc_5k_video7040_phase0
  val_bdd_pc_5k_video8125_phase1
  val_bdd_pc_5k_video8139_phase2
  val_bdd_pc_5k_video9450_phase4
)

echo "=== LTX 13B official detailer -> color-matched a10 BDD4 gate ==="
echo "Reuses 7040 and generates only three additional clips."
/workspace/wan_ticft/evaluation/check_gpu_free.sh "$GPU_ID" 36000
for path in "$MODEL" "$LORA" "$MANIFEST" "$BASE_REPORT" "$BDD1/detailer_s100/val_bdd_pc_5k_video7040_phase0.mp4"; do
  [[ -s "$path" ]] || { echo "Missing input: $path" >&2; exit 2; }
done

rm -rf "$OUT" "$BLEND"
mkdir -p "$OUT/detailer_s100"
cp "$BDD1/detailer_s100/val_bdd_pc_5k_video7040_phase0.mp4" "$OUT/detailer_s100/"

SAMPLE_ARGS=()
for sample_id in "${SAMPLES[@]}"; do
  SAMPLE_ARGS+=(--sample_id "$sample_id")
done

echo "[1/5] Generate/reuse detailer s100"
CUDA_VISIBLE_DEVICES="$GPU_ID" \
HF_HOME=/root/.cache/huggingface \
PYTHONPATH="/workspace:$REPO/src" \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
TOKENIZERS_PARALLELISM=false \
"$TRAIN_PYTHON" -m wan_ticft.evaluation.run_ltx13b_ic_lora_detailer_smoke \
  --manifest "$MANIFEST" \
  --source_report "$BASE_REPORT" \
  --model_path "$MODEL" \
  --lora_path "$LORA" \
  --out_dir "$OUT" \
  "${SAMPLE_ARGS[@]}" \
  --lora_scale 1.00 \
  --width 512 \
  --height 288 \
  --frames 41 \
  --steps 20 \
  --guidance 3.5 \
  --seed 42

echo "[2/5] Color-match and blend 10% detailer residual"
"$PYTHON" -m wan_ticft.evaluation.materialize_detailer_residual_blends \
  --base_report "$BASE_REPORT" \
  --detailer_report "$OUT/detailer_s100/report.csv" \
  --out_dir "$BLEND" \
  "${SAMPLE_ARGS[@]}" \
  --alpha 0.10

echo "[3/5] Pixel metrics"
"$PYTHON" -m wan_ticft.evaluation.evaluate_video_reports \
  --manifest "$OUT/selected_manifest.jsonl" \
  --reports \
    actor_depth_a125="$BASE_REPORT" \
    detailer_s100="$OUT/detailer_s100/report.csv" \
    cmblend_a10="$BLEND/cmblend_a10/report.csv" \
  --include_repeat_last \
  --offset_mode timestamp \
  --out_csv "${BLEND}_eval.csv"

echo "[4/5] FID/FVD proxy"
CUDA_VISIBLE_DEVICES="$GPU_ID" TORCH_HOME="$ROOT/tools/torch_cache" \
"$PYTHON" -m wan_ticft.evaluation.audit_perceptual_video_distribution \
  --manifest "$OUT/selected_manifest.jsonl" \
  --report actor_depth_a125="$BASE_REPORT" \
  --report detailer_s100="$OUT/detailer_s100/report.csv" \
  --report cmblend_a10="$BLEND/cmblend_a10/report.csv" \
  --out_csv "${BLEND}_proxy.csv" \
  --clip_frames 16 \
  --frame_stride 4

echo "[5/5] Contact sheets"
"$PYTHON" -m wan_ticft.evaluation.make_report_comparison_sheet \
  --manifest "$OUT/selected_manifest.jsonl" \
  --reports \
    actor_depth_a125="$BASE_REPORT" \
    detailer_s100="$OUT/detailer_s100/report.csv" \
    cmblend_a10="$BLEND/cmblend_a10/report.csv" \
  "${SAMPLE_ARGS[@]}" \
  --out_dir "${BLEND}_compare_sheets" \
  --frames 8 \
  --tile_width 260

echo
echo "eval=${BLEND}_eval.csv"
echo "proxy=${BLEND}_proxy.csv"
echo "sheets=${BLEND}_compare_sheets"
echo "Scale only if cmblend_a10 improves both aggregate FID/FVD and preserves pixel metrics across BDD4."

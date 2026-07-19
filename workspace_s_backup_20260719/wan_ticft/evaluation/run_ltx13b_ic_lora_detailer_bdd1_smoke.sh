#!/usr/bin/env bash
set -euo pipefail

cd /workspace

PYTHON=/workspace/experiments/wan_ticft/ltxv_trainer_env/bin/python
HF=/opt/conda/envs/wan22/bin/hf
ROOT=/workspace/experiments/wan_ticft
REPO=/workspace/third_party/LTX-Video-Trainer
MODEL_DIR=/workspace/models/ltx13b098_ic_detailer
MODEL="$MODEL_DIR/ltxv-13b-0.9.8-dev.safetensors"
LORA="$MODEL_DIR/ltxv-098-ic-lora-detailer-diffusers.safetensors"
MANIFEST="$ROOT/ltxv_bdd25_gate/dense40_manifest.jsonl"
SOURCE_REPORT="$ROOT/sa2va_actor_depth_strength_extension_bdd24/actor_depth_a125/report.csv"
OUT="$ROOT/ltx13b_ic_lora_detailer_bdd1_smoke"
GPU_ID="${CUDA_VISIBLE_DEVICES:-0}"
GPU_ID="${GPU_ID%%,*}"
SAMPLE=val_bdd_pc_5k_video7040_phase0

echo "=== Official LTX 13B 0.9.8 IC-LoRA Detailer BDD1 smoke ==="
echo "Reference=$SOURCE_REPORT"
echo "CPU offload is enabled; no submission folder is modified."
/workspace/wan_ticft/evaluation/check_gpu_free.sh "$GPU_ID" 36000
FREE_GIB=$(df --output=avail -BG /workspace | tail -1 | tr -dc '0-9')
if (( FREE_GIB < 45 )); then
  echo "Need at least 45 GiB free on /workspace; found ${FREE_GIB} GiB." >&2
  exit 2
fi
for path in "$MANIFEST" "$SOURCE_REPORT"; do
  [[ -s "$path" ]] || { echo "Missing input: $path" >&2; exit 2; }
done
mkdir -p "$MODEL_DIR"

if [[ ! -s "$MODEL" ]]; then
  echo "[download] LTX 13B 0.9.8 DEV (~28.6 GB)"
  "$HF" download Lightricks/LTX-Video ltxv-13b-0.9.8-dev.safetensors \
    --local-dir "$MODEL_DIR"
fi
if [[ ! -s "$LORA" ]]; then
  echo "[download] Official IC-LoRA Detailer (~1.3 GB)"
  "$HF" download Lightricks/LTX-Video-ICLoRA-detailer-13b-0.9.8 \
    ltxv-098-ic-lora-detailer-diffusers.safetensors \
    --local-dir "$MODEL_DIR"
fi

rm -rf "$OUT"
echo "[1/4] Generate detailer scales 0.50 and 1.00"
CUDA_VISIBLE_DEVICES="$GPU_ID" \
HF_HOME=/root/.cache/huggingface \
PYTHONPATH="/workspace:$REPO/src" \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
TOKENIZERS_PARALLELISM=false \
"$PYTHON" -m wan_ticft.evaluation.run_ltx13b_ic_lora_detailer_smoke \
  --manifest "$MANIFEST" \
  --source_report "$SOURCE_REPORT" \
  --model_path "$MODEL" \
  --lora_path "$LORA" \
  --out_dir "$OUT" \
  --sample_id "$SAMPLE" \
  --lora_scale 0.50 \
  --lora_scale 1.00 \
  --width 512 \
  --height 288 \
  --frames 41 \
  --steps 20 \
  --guidance 3.5 \
  --seed 42

echo "[2/4] Pixel metrics"
/opt/conda/envs/wan22/bin/python -m wan_ticft.evaluation.evaluate_video_reports \
  --manifest "$OUT/selected_manifest.jsonl" \
  --reports \
    actor_depth_a125="$SOURCE_REPORT" \
    detailer_s050="$OUT/detailer_s050/report.csv" \
    detailer_s100="$OUT/detailer_s100/report.csv" \
  --include_repeat_last \
  --offset_mode timestamp \
  --out_csv "${OUT}_eval.csv"

echo "[3/4] FID/FVD proxy"
CUDA_VISIBLE_DEVICES="$GPU_ID" TORCH_HOME="$ROOT/tools/torch_cache" \
/opt/conda/envs/wan22/bin/python -m wan_ticft.evaluation.audit_perceptual_video_distribution \
  --manifest "$OUT/selected_manifest.jsonl" \
  --report actor_depth_a125="$SOURCE_REPORT" \
  --report detailer_s050="$OUT/detailer_s050/report.csv" \
  --report detailer_s100="$OUT/detailer_s100/report.csv" \
  --out_csv "${OUT}_proxy.csv" \
  --clip_frames 16 \
  --frame_stride 4

echo "[4/4] Contact sheet"
/opt/conda/envs/wan22/bin/python -m wan_ticft.evaluation.make_report_comparison_sheet \
  --manifest "$OUT/selected_manifest.jsonl" \
  --reports \
    actor_depth_a125="$SOURCE_REPORT" \
    detailer_s050="$OUT/detailer_s050/report.csv" \
    detailer_s100="$OUT/detailer_s100/report.csv" \
  --sample_id "$SAMPLE" \
  --out_dir "${OUT}_compare_sheets" \
  --frames 8 \
  --tile_width 300

echo
echo "eval=${OUT}_eval.csv"
echo "proxy=${OUT}_proxy.csv"
echo "sheets=${OUT}_compare_sheets"
echo "Reject if motion/geometry changes, or unless one scale improves both FID and FVD with tolerable pixel loss."

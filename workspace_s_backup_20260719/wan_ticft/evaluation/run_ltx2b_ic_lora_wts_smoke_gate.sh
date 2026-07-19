#!/usr/bin/env bash
set -euo pipefail

cd /workspace

ROOT=/workspace/experiments/wan_ticft
DATA="$ROOT/ltx2b_ic_lora_wts_ip_data"
SMOKE="$ROOT/ltx2b_ic_lora_wts_ip_smoke"
OUT="$ROOT/ltx2b_ic_lora_wts_ip_smoke_gate"
GPU_ID="${CUDA_VISIBLE_DEVICES:-2}"
GPU_ID="${GPU_ID%%,*}"

echo "=== Score existing LTX 2B IC-LoRA WTS IP smoke ==="
echo "This does not train or generate videos. It scores the two existing 9-frame validation samples."

for path in "$DATA/validation.json" "$DATA/val_region_base" "$SMOKE/samples"; do
  [[ -e "$path" ]] || { echo "Missing input: $path" >&2; exit 2; }
done

rm -rf "$OUT"
CUDA_VISIBLE_DEVICES="$GPU_ID" /opt/conda/envs/wan22/bin/python \
  -m wan_ticft.evaluation.score_ltx_ic_lora_smoke \
  --validation_json "$DATA/validation.json" \
  --region_base "$DATA/val_region_base" \
  --samples "$SMOKE/samples" \
  --out_dir "$OUT"

echo
echo "Gate rule: close this direction unless step100 or step200 beats region_medium"
echo "on both FID and FVD proxy without losing more than 0.10 PSNR or 0.002 SSIM."
echo "Two clips are only a rejection gate; passing would require inference on all val8."

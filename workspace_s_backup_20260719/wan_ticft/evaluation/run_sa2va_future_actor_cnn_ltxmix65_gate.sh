#!/usr/bin/env bash
set -euo pipefail

cd /workspace

PYTHON=/opt/conda/envs/wan22/bin/python
ROOT=/workspace/experiments/wan_ticft
MANIFEST="$ROOT/ltxv_bdd25_gate/dense40_manifest.jsonl"
BASE_REPORT="$ROOT/cnn_selector_perceptual_tradeoff_gate/cnn_ltxmix65/report.csv"
OUT="$ROOT/sa2va4b_future_actor_cnn_ltxmix65_gate3"
MODEL="${SA2VA_MODEL:-ByteDance/Sa2VA-4B}"
MODEL_REVISION="${SA2VA_REVISION:-3fee777d49ee9276eac51ea3e5f9b69e81d09be6}"
GPU_ID="${CUDA_VISIBLE_DEVICES:-2}"
GPU_ID="${GPU_ID%%,*}"

echo "=== Sa2VA actor tracking: real anchor -> cnn_ltxmix65 future ==="
echo "Audit only: one real anchor plus nine uniformly sampled prediction frames."
/workspace/wan_ticft/evaluation/check_gpu_free.sh "$GPU_ID" 22000
[[ -s "$MANIFEST" ]] || { echo "Missing manifest: $MANIFEST" >&2; exit 2; }
[[ -s "$BASE_REPORT" ]] || { echo "Missing report: $BASE_REPORT" >&2; exit 2; }
"$PYTHON" -c 'import timm, peft; print("dependencies=ok")'

rm -rf "$OUT"
CUDA_VISIBLE_DEVICES="$GPU_ID" \
HF_HOME=/root/.cache/huggingface \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
TOKENIZERS_PARALLELISM=false \
"$PYTHON" -m wan_ticft.evaluation.run_sa2va_future_actor_gate \
  --manifest "$MANIFEST" \
  --base_report "$BASE_REPORT" \
  --out_dir "$OUT" \
  --model "$MODEL" \
  --revision "$MODEL_REVISION" \
  --sample_id val_bdd_pc_5k_video2556_phase2 \
  --sample_id val_bdd_pc_5k_video8125_phase1 \
  --sample_id val_bdd_pc_5k_video8139_phase2 \
  --future_samples 9

echo
echo "report=$OUT/report.csv"
echo "sheets=$OUT/*/future_segmentation_sheet.jpg"
echo
echo "Gate: continue only if the anchor is correct and the same physical pedestrian is followed"
echo "through generated frames without switching to cars, other people, shadows, or artifacts."
echo "It is valid for masks to become empty only after the actor visibly leaves the frame."

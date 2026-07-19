#!/usr/bin/env bash
set -euo pipefail

cd /workspace

PYTHON=/opt/conda/envs/wan22/bin/python
PIP=/opt/conda/envs/wan22/bin/pip
ROOT=/workspace/experiments/wan_ticft
MANIFEST="$ROOT/ltxv_bdd25_gate/dense40_manifest.jsonl"
OUT="$ROOT/sa2va4b_caption_actor_bdd4"
MODEL="${SA2VA_MODEL:-ByteDance/Sa2VA-4B}"
MODEL_REVISION="${SA2VA_REVISION:-3fee777d49ee9276eac51ea3e5f9b69e81d09be6}"
GPU_ID="${CUDA_VISIBLE_DEVICES:-2}"
GPU_ID="${GPU_ID%%,*}"

echo "=== Sa2VA caption-referred actor segmentation: BDD4 ==="
echo "Segmentation audit only; this does not generate, inpaint, or modify submission frames."
echo "model=$MODEL"
echo "revision=$MODEL_REVISION"
/workspace/wan_ticft/evaluation/check_gpu_free.sh "$GPU_ID" 22000
[[ -s "$MANIFEST" ]] || { echo "Missing manifest: $MANIFEST" >&2; exit 2; }

"$PYTHON" - <<'PY' || "$PIP" install --no-cache-dir 'peft>=0.12,<0.18'
import peft
print(f"peft={peft.__version__}")
PY

"$PYTHON" - <<'PY' || "$PIP" install --no-cache-dir 'timm==1.0.12'
import timm
print(f"timm={timm.__version__}")
PY

rm -rf "$OUT"
CUDA_VISIBLE_DEVICES="$GPU_ID" \
HF_HOME=/root/.cache/huggingface \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
TOKENIZERS_PARALLELISM=false \
"$PYTHON" -m wan_ticft.evaluation.run_sa2va_caption_actor_bdd4 \
  --manifest "$MANIFEST" \
  --out_dir "$OUT" \
  --model "$MODEL" \
  --revision "$MODEL_REVISION" \
  --sample_id val_bdd_pc_5k_video2556_phase2 \
  --sample_id val_bdd_pc_5k_video8125_phase1 \
  --sample_id val_bdd_pc_5k_video8139_phase2 \
  --sample_id val_bdd_pc_5k_video8277_phase2 \
  --limit 4

echo
echo "report=$OUT/report.csv"
echo "sheets=$OUT/*/segmentation_sheet.jpg"
echo
echo "Gate: continue only if the caption pedestrian is the only selected actor in at least 3/4 sheets,"
echo "the mask follows the same body through all four frames, and background coverage is negligible."
echo "Do not build inpainting/compositing until this visual gate passes."

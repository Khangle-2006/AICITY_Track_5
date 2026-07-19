#!/usr/bin/env bash
set -euo pipefail

cd /workspace

PYTHON=/opt/conda/envs/wan22/bin/python
ROOT=/workspace/experiments/wan_ticft
MANIFEST="$ROOT/ltxv_bdd25_gate/dense40_manifest.jsonl"
OUT="$ROOT/qwen25vl_caption_grounding_bdd4"
MODEL=Qwen/Qwen2.5-VL-7B-Instruct
GPU_ID="${CUDA_VISIBLE_DEVICES:-2}"
GPU_ID="${GPU_ID%%,*}"
MAX_PIXELS="${VLM_MAX_PIXELS:-401408}"

echo "=== Qwen2.5-VL caption-grounded actor audit: BDD4 ==="
echo "Audit only: four history frames + challenge caption -> actor boxes and motion labels."
/workspace/wan_ticft/evaluation/check_gpu_free.sh "$GPU_ID" 26000

[[ -s "$MANIFEST" ]] || { echo "Missing manifest: $MANIFEST" >&2; exit 2; }
"$PYTHON" - <<'PY'
import transformers
from transformers import Qwen2_5_VLForConditionalGeneration
assert tuple(int(x) for x in transformers.__version__.split('.')[:2]) >= (4, 49), transformers.__version__
print(f"transformers={transformers.__version__}")
PY

rm -rf "$OUT"
CUDA_VISIBLE_DEVICES="$GPU_ID" \
HF_HOME=/root/.cache/huggingface \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
"$PYTHON" -m wan_ticft.evaluation.run_qwen25vl_caption_grounding \
  --manifest "$MANIFEST" \
  --out_dir "$OUT" \
  --model "$MODEL" \
  --sample_id val_bdd_pc_5k_video2556_phase2 \
  --sample_id val_bdd_pc_5k_video8125_phase1 \
  --sample_id val_bdd_pc_5k_video8139_phase2 \
  --sample_id val_bdd_pc_5k_video8277_phase2 \
  --limit 4 \
  --max_pixels "$MAX_PIXELS" \
  --max_new_tokens 700

echo
echo "report=$OUT/report.csv"
echo "sheets=$OUT/*/grounding_sheet.jpg"
echo
echo "Gate: continue only if the caption actor is correctly grounded in at least 3/4 sheets,"
echo "parked vehicles are usually labeled stationary, and ego motion is plausible."
echo "Do not integrate boxes into rendering before this visual gate passes."

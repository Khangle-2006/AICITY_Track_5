#!/usr/bin/env bash
set -euo pipefail

cd /workspace

PYTHON=/opt/conda/envs/wan22/bin/python
REPO=/workspace/third_party/LTX-Video-Trainer
ENV=/workspace/experiments/wan_ticft/ltx_lora_trainer_env
DATA=/workspace/experiments/wan_ticft/ltx_ic_lora_bdd_smoke20
MANIFEST=/workspace/experiments/wan_ticft/manifests/train_ticft_balanced_121f_frames.jsonl
GPU_ID="${CUDA_VISIBLE_DEVICES:-0}"
GPU_ID="${GPU_ID%%,*}"

echo "=== Materialize 20 paired IC-LoRA clips ==="
"$PYTHON" -m wan_ticft.evaluation.materialize_ltx_ic_lora_smoke \
  --manifest "$MANIFEST" \
  --out_dir "$DATA" \
  --per_phase 4 \
  --frames 121 \
  --width 512 \
  --height 288 \
  --clean

echo
echo "=== Verify paired videos ==="
DATA="$DATA" "$PYTHON" - <<'PY'
import json
import os
from pathlib import Path
import cv2

root = Path(os.environ["DATA"])
rows = [json.loads(line) for line in (root / "dataset.jsonl").read_text().splitlines() if line.strip()]
bad = []
for row in rows:
    counts = []
    for key in ("media_path", "reference_path"):
        cap = cv2.VideoCapture(str(root / row[key]))
        counts.append(int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0))
        cap.release()
    if counts != [121, 121]:
        bad.append((row, counts))
print({"rows": len(rows), "bad": len(bad), "first_bad": bad[:1]})
if len(rows) != 20 or bad:
    raise SystemExit("paired video verification failed")
PY

echo
echo "=== Preprocess targets + references ==="
/workspace/wan_ticft/evaluation/check_gpu_free.sh "$GPU_ID" 20000
if [[ ! -x "$ENV/bin/python" ]]; then
  echo "Missing trainer environment: $ENV" >&2
  exit 1
fi
rm -rf "$DATA/.precomputed"
cd "$REPO"
TOKENIZERS_PARALLELISM=false "$ENV/bin/python" scripts/preprocess_dataset.py \
  "$DATA/dataset.jsonl" \
  --resolution-buckets "512x288x121" \
  --caption-column "caption" \
  --video-column "media_path" \
  --reference-column "reference_path" \
  --batch-size 1 \
  --device cuda \
  --vae-tiling \
  --model-source "LTXV_2B_0.9.6_DISTILLED"

echo
echo "=== Verify IC-LoRA caches ==="
DATA="$DATA" "$ENV/bin/python" - <<'PY'
import os
from pathlib import Path

root = Path(os.environ["DATA"]) / ".precomputed"
counts = {
    name: len(list((root / name).rglob("*.pt")))
    for name in ("latents", "conditions", "reference_latents")
}
print(counts)
if counts != {"latents": 20, "conditions": 20, "reference_latents": 20}:
    raise SystemExit("expected exactly 20 files in every IC-LoRA cache")
PY

echo
echo "PASS: IC-LoRA paired dataset and caches are ready."

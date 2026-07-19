#!/usr/bin/env bash
set -euo pipefail

cd /workspace

PYTHON=/opt/conda/envs/wan22/bin/python
MANIFEST=/workspace/experiments/wan_ticft/manifests/train_ticft_balanced_121f_frames.jsonl
OUT=/workspace/experiments/wan_ticft/ltx_lora_bdd_smoke20

echo "=== Materialize LTX BDD LoRA smoke dataset ==="
"$PYTHON" -m wan_ticft.evaluation.materialize_ltx_lora_smoke \
  --manifest "$MANIFEST" \
  --out_dir "$OUT" \
  --per_phase 4 \
  --frames 121 \
  --clean

echo
echo "=== Verify clips ==="
"$PYTHON" - <<'PY'
import cv2
import json
from pathlib import Path

root = Path("/workspace/experiments/wan_ticft/ltx_lora_bdd_smoke20")
rows = [json.loads(line) for line in (root / "dataset.jsonl").read_text().splitlines() if line.strip()]
bad = []
for row in rows:
    path = Path(row["media_path"])
    cap = cv2.VideoCapture(str(path))
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    cap.release()
    if frames != 121 or width <= 0 or height <= 0:
        bad.append((str(path), frames, width, height))
print({"rows": len(rows), "bad": len(bad), "first_bad": bad[:1]})
if len(rows) != 20 or bad:
    raise SystemExit(1)
PY

echo
echo "=== Prepare result ==="
echo "Dataset: $OUT/dataset.jsonl"
echo "Trainer config: /workspace/wan_ticft/evaluation/ltxv_2b_lora_bdd_smoke.yaml"
echo "No preprocessing or training was launched."
echo "Next gate: install trainer environment, then preprocess these 20 clips at 512x288x121."

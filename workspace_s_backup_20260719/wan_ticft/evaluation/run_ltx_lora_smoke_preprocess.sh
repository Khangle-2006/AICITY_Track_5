#!/usr/bin/env bash
set -euo pipefail

cd /workspace

REPO=/workspace/third_party/LTX-Video-Trainer
ENV=/workspace/experiments/wan_ticft/ltx_lora_trainer_env
DATA=/workspace/experiments/wan_ticft/ltx_lora_bdd_smoke20
TOOLS=/workspace/experiments/wan_ticft/tools
GPU_ID="${CUDA_VISIBLE_DEVICES:-0}"
GPU_ID="${GPU_ID%%,*}"

echo "=== LTX LoRA smoke preprocessing gate ==="
/workspace/wan_ticft/evaluation/check_gpu_free.sh "$GPU_ID" 20000
df -h /workspace

if [[ ! -d "$REPO/.git" ]]; then
  echo "Missing trainer repo: $REPO" >&2
  exit 1
fi
if [[ ! -f "$DATA/dataset.jsonl" ]]; then
  echo "Missing smoke dataset: $DATA/dataset.jsonl" >&2
  exit 1
fi

echo
echo "=== Install official trainer environment ==="
UV_BIN="$(command -v uv || true)"
if [[ -z "$UV_BIN" && -x "$TOOLS/bin/uv" ]]; then
  UV_BIN="$TOOLS/bin/uv"
fi
if [[ -z "$UV_BIN" ]]; then
  echo "uv not found; installing a workspace-local copy."
  mkdir -p "$TOOLS"
  curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR="$TOOLS/bin" sh
  UV_BIN="$TOOLS/bin/uv"
fi
echo "uv: $("$UV_BIN" --version)"
UV_PROJECT_ENVIRONMENT="$ENV" "$UV_BIN" sync --project "$REPO" --no-dev

echo
echo "=== Preprocess 20 clips at 512x288x121 ==="
DATA="$DATA" "$ENV/bin/python" - <<'PY'
import json
import os
from pathlib import Path

root = Path(os.environ["DATA"])
dataset = root / "dataset.jsonl"
rows = [json.loads(line) for line in dataset.read_text(encoding="utf-8").splitlines() if line.strip()]
for row in rows:
    path = Path(row["media_path"])
    if path.is_absolute():
        row["media_path"] = str(path.relative_to(root))
dataset.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")

# Older trainer behavior writes caption embeddings beside videos when metadata
# contains absolute paths. Remove those misplaced cache files.
for path in (root / "videos").glob("*.pt"):
    path.unlink()
print({"metadata_rows": len(rows), "relative_media_paths": sum(not Path(row["media_path"]).is_absolute() for row in rows)})
PY
rm -rf "$DATA/.precomputed"
cd "$REPO"
TOKENIZERS_PARALLELISM=false "$ENV/bin/python" scripts/preprocess_dataset.py \
  "$DATA/dataset.jsonl" \
  --resolution-buckets "512x288x121" \
  --caption-column "caption" \
  --video-column "media_path" \
  --batch-size 1 \
  --device cuda \
  --vae-tiling \
  --model-source "LTXV_2B_0.9.6_DISTILLED"

echo
echo "=== Verify precomputed dataset ==="
"$ENV/bin/python" - <<'PY'
from pathlib import Path

root = Path("/workspace/experiments/wan_ticft/ltx_lora_bdd_smoke20/.precomputed")
counts = {
    "latents": len(list((root / "latents").rglob("*.pt"))),
    "conditions": len(list((root / "conditions").rglob("*.pt"))),
}
print(counts)
if counts["latents"] != 20 or counts["conditions"] != 20:
    raise SystemExit("expected exactly 20 latents and 20 conditions")
PY

echo
echo "=== Preprocessing gate result ==="
echo "PASS: 20 clips cached."
echo "No training was launched."
echo "Next gate: run the 50-step LoRA smoke and compare against unmodified LTX on a small BDD validation set."

#!/usr/bin/env bash
set -euo pipefail

cd /workspace

PYTHON=${PYTHON:-/opt/conda/envs/wan22/bin/python}
ROOT=${ROOT:-/workspace/experiments/wan_ticft}
REPO=${REPO:-/workspace/third_party/LTX-Video-Trainer}
ENV=${ENV:-$ROOT/ltx_lora_trainer_env}
TOOLS=${TOOLS:-$ROOT/tools}
MANIFEST=${MANIFEST:-$ROOT/manifests/train_ticft_balanced_121f_frames.jsonl}
VAL_MANIFEST=${VAL_MANIFEST:-$ROOT/manifests/val_ticft_balanced_125_frames.jsonl}
DATA=${DATA:-$ROOT/ltx2b_camera_style_lora_bdd100}
CONFIG=${CONFIG:-/workspace/wan_ticft/evaluation/ltxv_2b_camera_style_lora_bdd100_s300.yaml}
OUT=${OUT:-$ROOT/ltx2b_camera_style_lora_bdd100_output}
EVAL_OUT=${EVAL_OUT:-$ROOT/ltx2b_camera_style_lora_bdd100_eval4}
PER_PHASE=${PER_PHASE:-${LORA_PER_PHASE:-20}}
FRAMES=${FRAMES:-121}
REUSE_DATA=${REUSE_DATA:-1}
REUSE_PRECOMPUTED=${REUSE_PRECOMPUTED:-1}
GPU_ID="${CUDA_VISIBLE_DEVICES:-3}"
GPU_ID="${GPU_ID%%,*}"

echo "=== LTX2B camera-style LoRA BDD100 s300 ==="
echo "data=$DATA"
echo "config=$CONFIG"
echo "out=$OUT"
echo "gpu=$GPU_ID per_phase=$PER_PHASE frames=$FRAMES"

for path in "$MANIFEST" "$VAL_MANIFEST" "$CONFIG"; do
  [[ -s "$path" ]] || { echo "Missing input: $path" >&2; exit 2; }
done
[[ -d "$REPO" ]] || { echo "Missing trainer repo: $REPO" >&2; exit 2; }

/workspace/wan_ticft/evaluation/check_gpu_free.sh "$GPU_ID" 20000

echo "[1/6] Materialize 100 real BDD clips"
EXPECTED_ROWS=$((PER_PHASE * 5))
if [[ "$REUSE_DATA" == "1" && -s "$DATA/dataset.jsonl" ]]; then
  DATA_ROWS="$("$PYTHON" - "$DATA/dataset.jsonl" <<'PY'
import sys
from pathlib import Path
path = Path(sys.argv[1])
print(sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip()))
PY
)"
  if [[ "$DATA_ROWS" == "$EXPECTED_ROWS" ]]; then
    echo "Reusing existing dataset: $DATA/dataset.jsonl ($DATA_ROWS rows)"
  else
    echo "Existing dataset has $DATA_ROWS rows, expected $EXPECTED_ROWS; rematerializing"
    "$PYTHON" -m wan_ticft.evaluation.materialize_ltx_lora_smoke \
      --manifest "$MANIFEST" \
      --out_dir "$DATA" \
      --per_phase "$PER_PHASE" \
      --frames "$FRAMES" \
      --clean
  fi
else
  "$PYTHON" -m wan_ticft.evaluation.materialize_ltx_lora_smoke \
    --manifest "$MANIFEST" \
    --out_dir "$DATA" \
    --per_phase "$PER_PHASE" \
    --frames "$FRAMES" \
    --clean
fi

echo "[2/6] Install/reuse trainer environment"
UV_BIN="$(command -v uv || true)"
if [[ -z "$UV_BIN" && -x "$TOOLS/bin/uv" ]]; then
  UV_BIN="$TOOLS/bin/uv"
fi
if [[ -z "$UV_BIN" ]]; then
  mkdir -p "$TOOLS"
  curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR="$TOOLS/bin" sh
  UV_BIN="$TOOLS/bin/uv"
fi
echo "uv: $("$UV_BIN" --version)"
NEED_SYNC=1
if [[ -x "$ENV/bin/python" ]]; then
  ENV_PY_VERSION="$("$ENV/bin/python" - <<'PY'
import sys
print(f"{sys.version_info.major}.{sys.version_info.minor}")
PY
)"
  if [[ "$ENV_PY_VERSION" == "3.10" ]]; then
    NEED_SYNC=0
    echo "Reusing trainer env: $ENV (python $ENV_PY_VERSION)"
  else
    echo "Trainer env uses python $ENV_PY_VERSION; recreating with $PYTHON"
  fi
fi
if [[ "$NEED_SYNC" == "1" ]]; then
  rm -rf "$ENV"
  UV_PROJECT_ENVIRONMENT="$ENV" UV_PYTHON="$PYTHON" "$UV_BIN" sync --project "$REPO" --no-dev --python "$PYTHON"
fi
"$ENV/bin/python" --version

echo "[3/6] Normalize metadata paths"
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
for path in (root / "videos").glob("*.pt"):
    path.unlink()
print({"metadata_rows": len(rows)})
PY

echo "[4/6] Preprocess latents 512x288x121"
PRECOMPUTED_READY=0
if [[ "$REUSE_PRECOMPUTED" == "1" && -d "$DATA/.precomputed" ]]; then
  LATENT_COUNT="$(find "$DATA/.precomputed" -type f -name '*latent*.pt' 2>/dev/null | wc -l | tr -d ' ')"
  if [[ "$LATENT_COUNT" -ge "$EXPECTED_ROWS" ]]; then
    PRECOMPUTED_READY=1
    echo "Reusing precomputed latents: $DATA/.precomputed ($LATENT_COUNT latent files)"
  else
    echo "Precomputed latents incomplete ($LATENT_COUNT/$EXPECTED_ROWS); rebuilding"
  fi
fi
cd "$REPO"
if [[ "$PRECOMPUTED_READY" != "1" ]]; then
  rm -rf "$DATA/.precomputed"
  TOKENIZERS_PARALLELISM=false "$ENV/bin/python" scripts/preprocess_dataset.py \
    "$DATA/dataset.jsonl" \
    --resolution-buckets "512x288x121" \
    --caption-column "caption" \
    --video-column "media_path" \
    --batch-size 1 \
    --device cuda \
    --vae-tiling \
    --model-source "LTXV_2B_0.9.6_DISTILLED"
fi

echo "[5/6] Train LoRA"
rm -rf "$OUT"
cd "$REPO"
TOKENIZERS_PARALLELISM=false \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
"$ENV/bin/python" scripts/train.py "$CONFIG"

echo "[6/6] Eval latest checkpoint on BDD4 smoke"
LATEST="$(find "$OUT/checkpoints" -maxdepth 1 -type f -name 'lora_weights_step_*.safetensors' | sort | tail -n 1)"
[[ -s "$LATEST" ]] || { echo "No LoRA checkpoint found in $OUT/checkpoints" >&2; exit 2; }
echo "latest=$LATEST"
rm -rf "$EVAL_OUT"
PYTHONPATH="/workspace:$REPO/src" TOKENIZERS_PARALLELISM=false \
"$ENV/bin/python" -m wan_ticft.evaluation.evaluate_ltx_lora_smoke \
  --manifest "$VAL_MANIFEST" \
  --checkpoint "$LATEST" \
  --out_dir "$EVAL_OUT" \
  --limit 4

PYTHONPATH="/workspace:$REPO/src" "$PYTHON" -m wan_ticft.evaluation.evaluate_video_reports \
  --manifest "$EVAL_OUT/selected_manifest.jsonl" \
  --reports base="$EVAL_OUT/base_report.csv" lora="$EVAL_OUT/lora_report.csv" \
  --include_repeat_last \
  --offset_mode timestamp \
  --limit 4 \
  --out_csv "$EVAL_OUT/eval_sparse.csv"

echo
echo "checkpoint=$LATEST"
echo "eval=$EVAL_OUT/eval_sparse.csv"
echo "videos=$EVAL_OUT"
echo "Gate: continue only if LoRA makes base less synthetic without geometry/timing collapse."

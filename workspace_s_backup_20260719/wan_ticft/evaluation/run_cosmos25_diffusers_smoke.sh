#!/usr/bin/env bash
set -euo pipefail

cd /workspace

REPO=/workspace/third_party/cosmos-predict2.5
ROOT=/workspace/experiments/wan_ticft
ENV_DIR="$ROOT/cosmos25_diffusers_env"
CACHE_DIR="$ROOT/cosmos25_hf_cache"
OUT_DIR="$ROOT/cosmos25_diffusers_smoke36"
INPUT="$ROOT/cosmos25_bdd_smoke4/history_videos/video135.mp4"
OUTPUT="$OUT_DIR/video135_cosmos25_17f_s36.mp4"
GPU_LIST="${CUDA_VISIBLE_DEVICES:-3}"
GPU_ID="$GPU_LIST"
GPU_ID="${GPU_ID%%,*}"
SECOND_GPU=""
if [[ "$GPU_LIST" == *,* ]]; then
  SECOND_GPU="${GPU_LIST#*,}"
  SECOND_GPU="${SECOND_GPU%%,*}"
fi

echo "=== Cosmos-Predict2.5 2B Diffusers Video2World smoke ==="
echo "Candidate: post-trained 2B, one BDD clip, 17 frames, 36 steps"
echo "GPUs: $GPU_LIST"

if [[ ! -f "$INPUT" ]]; then
  echo "Missing input: $INPUT" >&2
  exit 2
fi

/workspace/wan_ticft/evaluation/check_gpu_free.sh "$GPU_ID" 42000
if [[ -n "$SECOND_GPU" ]]; then
  /workspace/wan_ticft/evaluation/check_gpu_free.sh "$SECOND_GPU" 30000
fi

FREE_GIB=$(df --output=avail -BG /workspace | tail -1 | tr -dc '0-9')
if (( FREE_GIB < 80 )); then
  echo "Need at least 80 GiB free on /workspace; found ${FREE_GIB} GiB." >&2
  exit 2
fi

mkdir -p "$OUT_DIR" "$CACHE_DIR"

if [[ ! -x "$ENV_DIR/bin/python" ]]; then
  echo "Creating isolated environment: $ENV_DIR"
  /opt/conda/envs/wan22/bin/python -m venv "$ENV_DIR"
  "$ENV_DIR/bin/python" -m pip install --upgrade pip setuptools wheel
fi

if ! "$ENV_DIR/bin/python" - <<'PY' >/dev/null 2>&1
import accelerate
import diffusers
import imageio_ffmpeg
import torch
import transformers
import tyro
from diffusers import Cosmos2_5_PredictBasePipeline
PY
then
  echo "Installing bounded Diffusers inference dependencies."
  "$ENV_DIR/bin/python" -m pip install \
    --extra-index-url https://download.pytorch.org/whl/cu128 \
    "torch==2.10.0" \
    "transformers==4.57.1" \
    accelerate \
    tyro \
    pydantic \
    "imageio[ffmpeg]" \
    "diffusers @ git+https://github.com/huggingface/diffusers.git"
fi

if ! "$ENV_DIR/bin/python" - <<'PY' >/dev/null 2>&1
import peft
from cosmos_guardrail import CosmosSafetyChecker

assert peft.__version__ == "0.17.1"
PY
then
  echo "Installing Diffusers-compatible PEFT and Cosmos guardrail without dependency upgrades."
  "$ENV_DIR/bin/python" -m pip install --no-deps "peft==0.17.1" "cosmos_guardrail==0.1.0"
fi

echo
echo "Environment:"
"$ENV_DIR/bin/python" - <<'PY'
import diffusers
import torch
import transformers

print("torch", torch.__version__)
print("cuda", torch.version.cuda)
print("diffusers", diffusers.__version__)
print("transformers", transformers.__version__)
print("cuda_available", torch.cuda.is_available())
PY

rm -f "$OUTPUT"

PROMPT="Continue the observed forward-facing driving scene with physically consistent vehicle and pedestrian motion. Preserve the road layout, camera viewpoint, lighting, object identities, and sharp geometry."

echo
echo "Starting Cosmos smoke. Predictor and safety checker use separate devices."
START=$(date +%s)
set +e
CUDA_VISIBLE_DEVICES="$GPU_LIST" \
COSMOS_SAFETY_DEVICE="${COSMOS_SAFETY_DEVICE:-cpu}" \
HF_HUB_CACHE="$CACHE_DIR" \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
timeout 90m "$ENV_DIR/bin/python" "$REPO/scripts/diffusers_inference.py" \
  --input_path "$INPUT" \
  --prompt "$PROMPT" \
  --output_path "$OUTPUT" \
  --num_output_frames 17 \
  --num_steps 36 \
  --seed 1
STATUS=$?
set -e
ELAPSED=$(( $(date +%s) - START ))

echo
echo "elapsed_seconds=$ELAPSED"
nvidia-smi --query-gpu=index,name,memory.used,memory.free,utilization.gpu --format=csv,noheader

if (( STATUS != 0 )); then
  echo "Cosmos smoke failed with status $STATUS." >&2
  echo "Do not retry unless the error is specifically CUDA OOM or a small dependency mismatch." >&2
  exit "$STATUS"
fi

if [[ ! -s "$OUTPUT" ]]; then
  echo "Inference returned success but output is missing: $OUTPUT" >&2
  exit 3
fi

echo
echo "Output probe:"
"$ENV_DIR/bin/python" - "$OUTPUT" <<'PY'
import sys
import cv2

capture = cv2.VideoCapture(sys.argv[1])
if not capture.isOpened():
    raise RuntimeError(f"cannot decode {sys.argv[1]}")
print("width", int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)))
print("height", int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
print("fps", capture.get(cv2.CAP_PROP_FPS))
print("frames", int(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
capture.release()
PY

echo
echo "output=$OUTPUT"
echo
echo "Smoke rule:"
echo "  - Pass technical feasibility only if inference completes, output has 17 frames,"
echo "    and runtime/memory are practical."
echo "  - Then visually inspect identity, geometry, and motion before a four-sample gate."
echo "  - If single-GPU CUDA OOM occurs, allow one balanced two-A6000 correction only."

#!/usr/bin/env bash
set -euo pipefail

cd /workspace

ROOT=/workspace/experiments/wan_ticft
ENV_DIR="$ROOT/cosmos25_diffusers_env"
CACHE_DIR="$ROOT/framepack_hf_cache"
OUT_DIR="$ROOT/framepack_f1_smoke17"
INPUT_VIDEO="${INPUT_VIDEO:-$ROOT/cosmos25_bdd_smoke4/history_videos/video135.mp4}"
INPUT_IMAGE="$OUT_DIR/video135_last_history.png"
OUTPUT="$OUT_DIR/video135_framepack_f1_17f_s8.mp4"
GPU_ID="${CUDA_VISIBLE_DEVICES:-3}"
GPU_ID="${GPU_ID%%,*}"

echo "=== FramePack F1 Diffusers smoke ==="
echo "Model: lllyasviel/FramePack_F1_I2V_HY_20250503"
echo "Input video: $INPUT_VIDEO"
echo "GPU: $GPU_ID"

if [[ ! -x "$ENV_DIR/bin/python" ]]; then
  echo "Missing env: $ENV_DIR" >&2
  exit 2
fi
if [[ ! -f "$INPUT_VIDEO" ]]; then
  echo "Missing input video: $INPUT_VIDEO" >&2
  exit 2
fi

/workspace/wan_ticft/evaluation/check_gpu_free.sh "$GPU_ID" 42000

FREE_GIB=$(df --output=avail -BG /workspace | tail -1 | tr -dc '0-9')
if (( FREE_GIB < 70 )); then
  echo "Need at least 70 GiB free on /workspace; found ${FREE_GIB} GiB." >&2
  exit 2
fi

mkdir -p "$OUT_DIR" "$CACHE_DIR"

"$ENV_DIR/bin/python" - <<'PY'
import diffusers
import torch
import transformers
from diffusers import HunyuanVideoFramepackPipeline, HunyuanVideoFramepackTransformer3DModel

print("torch", torch.__version__)
print("diffusers", diffusers.__version__)
print("transformers", transformers.__version__)
print("has_framepack", HunyuanVideoFramepackPipeline.__name__, HunyuanVideoFramepackTransformer3DModel.__name__)
PY

/opt/conda/envs/wan22/bin/python - <<PY
from pathlib import Path

import cv2

video = Path("$INPUT_VIDEO")
out = Path("$INPUT_IMAGE")
cap = cv2.VideoCapture(str(video))
if not cap.isOpened():
    raise RuntimeError(f"cannot open {video}")
count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
index = max(0, count - 1)
cap.set(cv2.CAP_PROP_POS_FRAMES, index)
ok, frame = cap.read()
cap.release()
if not ok or frame is None:
    raise RuntimeError(f"cannot read last frame from {video}")
out.parent.mkdir(parents=True, exist_ok=True)
if not cv2.imwrite(str(out), frame):
    raise RuntimeError(f"cannot write {out}")
print(f"input_image={out} source_frame={index}")
PY

cat > "$OUT_DIR/run_framepack_f1_smoke.py" <<'PY'
from pathlib import Path

import torch
from diffusers import HunyuanVideoFramepackPipeline, HunyuanVideoFramepackTransformer3DModel
from diffusers.utils import export_to_video, load_image
from transformers import SiglipImageProcessor, SiglipVisionModel

out_dir = Path("/workspace/experiments/wan_ticft/framepack_f1_smoke17")
input_image = out_dir / "video135_last_history.png"
output = out_dir / "video135_framepack_f1_17f_s8.mp4"

prompt = (
    "Continue the same forward-facing driving camera scene for the next second. "
    "Preserve the exact road layout, dashboard, lighting, camera viewpoint, vehicles, "
    "pedestrians, signs, and object identities. Add only realistic near-future traffic motion."
)

transformer = HunyuanVideoFramepackTransformer3DModel.from_pretrained(
    "lllyasviel/FramePack_F1_I2V_HY_20250503",
    torch_dtype=torch.bfloat16,
)
feature_extractor = SiglipImageProcessor.from_pretrained(
    "lllyasviel/flux_redux_bfl",
    subfolder="feature_extractor",
)
image_encoder = SiglipVisionModel.from_pretrained(
    "lllyasviel/flux_redux_bfl",
    subfolder="image_encoder",
    torch_dtype=torch.float16,
)
pipe = HunyuanVideoFramepackPipeline.from_pretrained(
    "hunyuanvideo-community/HunyuanVideo",
    transformer=transformer,
    feature_extractor=feature_extractor,
    image_encoder=image_encoder,
    torch_dtype=torch.float16,
)
pipe.enable_model_cpu_offload()
pipe.vae.enable_tiling()

image = load_image(str(input_image))
frames = pipe(
    image=image,
    prompt=prompt,
    height=512,
    width=768,
    num_frames=17,
    num_inference_steps=8,
    guidance_scale=9.0,
    generator=torch.Generator().manual_seed(1),
    sampling_type="vanilla",
).frames[0]
export_to_video(frames, str(output), fps=16)
print(f"output={output}")
PY

rm -f "$OUTPUT"
START=$(date +%s)
set +e
CUDA_VISIBLE_DEVICES="$GPU_ID" \
HF_HUB_CACHE="$CACHE_DIR" \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
timeout 150m "$ENV_DIR/bin/python" "$OUT_DIR/run_framepack_f1_smoke.py"
STATUS=$?
set -e
ELAPSED=$(( $(date +%s) - START ))

echo "elapsed_seconds=$ELAPSED"
nvidia-smi --query-gpu=index,name,memory.used,memory.free,utilization.gpu --format=csv,noheader

if (( STATUS != 0 )); then
  echo "FramePack smoke failed with status $STATUS." >&2
  exit "$STATUS"
fi
if [[ ! -s "$OUTPUT" ]]; then
  echo "Missing output: $OUTPUT" >&2
  exit 3
fi

ffprobe -v error \
  -select_streams v:0 \
  -show_entries stream=codec_name,width,height,r_frame_rate,nb_frames,duration \
  -of default=noprint_wrappers=1 \
  "$OUTPUT"

echo "output=$OUTPUT"

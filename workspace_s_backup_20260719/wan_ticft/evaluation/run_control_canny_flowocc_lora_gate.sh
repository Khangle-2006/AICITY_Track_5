#!/usr/bin/env bash
set -euo pipefail

cd /workspace

GPU_ID="${CUDA_VISIBLE_DEVICES:-3}"
GPU_ID="${GPU_ID%%,*}"
ROOT=/workspace/experiments/wan_ticft
TRAINER=/workspace/third_party/LTX-Video-Trainer
ENV="$ROOT/ltxv_trainer_env"
RUN="${RUN:-$ROOT/ltx2b_control_canny_flowocc_lora_bdd_gtref_train45_s100}"
DATA="$RUN/ic_lora_data"
CONFIG="$DATA/config.yaml"

echo "=== Control Canny+Flow/Occlusion IC-LoRA gate ==="
echo "run=$RUN"
/workspace/wan_ticft/evaluation/check_gpu_free.sh "$GPU_ID" 42000

test -s "$DATA/dataset.json"
test -s "$DATA/validation.json"
test -s "$CONFIG"

if [[ ! -x "$ENV/bin/python" ]]; then
  echo "[setup] create trainer venv: $ENV"
  /opt/conda/envs/wan22/bin/python -m venv "$ENV"
fi
if [[ ! -x "$ENV/bin/uv" ]]; then
  "$ENV/bin/pip" install --no-cache-dir uv
fi
UV_PROJECT_ENVIRONMENT="$ENV" "$ENV/bin/uv" sync --project "$TRAINER"

export PATH="$ENV/bin:$PATH"

echo "[1/3] Precompute target/control latents"
cd "$TRAINER"
CUDA_VISIBLE_DEVICES="$GPU_ID" HF_HOME=/root/.cache/huggingface \
"$ENV/bin/python" scripts/preprocess_dataset.py "$DATA/dataset.json" \
  --resolution-buckets 512x288x9 \
  --caption-column caption \
  --video-column media_path \
  --reference-column reference_path \
  --model-source LTXV_2B_0.9.6_DEV \
  --output-dir "$DATA/.precomputed" \
  --batch-size 1

cd /workspace
latent_count="$(find "$DATA/.precomputed/latents" -type f -name '*.pt' | wc -l)"
condition_count="$(find "$DATA/.precomputed/conditions" -type f -name '*.pt' | wc -l)"
reference_count="$(find "$DATA/.precomputed/reference_latents" -type f -name '*.pt' | wc -l)"
echo "precomputed: latents=$latent_count conditions=$condition_count references=$reference_count"
if [[ "$latent_count" -lt 24 || "$latent_count" != "$condition_count" || "$latent_count" != "$reference_count" ]]; then
  echo "Incomplete precomputed dataset" >&2
  exit 2
fi

echo "[2/3] Train IC-LoRA"
rm -rf "$RUN/train_output"
cd "$TRAINER"
CUDA_VISIBLE_DEVICES="$GPU_ID" HF_HOME=/root/.cache/huggingface \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
"$ENV/bin/python" scripts/train.py "$CONFIG"

echo "[3/3] Done"
echo "samples=$RUN/train_output/samples"
echo "checkpoints=$RUN/train_output/checkpoints"

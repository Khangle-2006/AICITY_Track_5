#!/usr/bin/env bash
set -euo pipefail

cd /workspace

ROOT=/workspace/experiments/wan_ticft
TRAINER=/workspace/third_party/LTX-Video-Trainer
ENV="$ROOT/ltxv_trainer_env"
DATA="$ROOT/ltx2b_ic_lora_wts_ip_data"
CONFIG="$DATA/ic_lora_2b_ip_smoke.yaml"
TRAIN_OUTPUT="$ROOT/ltx2b_ic_lora_wts_ip_smoke"
GPU_ID="${CUDA_VISIBLE_DEVICES:-2}"
GPU_ID="${GPU_ID%%,*}"

echo "=== Resume LTX-Video 2B IC-LoRA WTS IP-camera smoke ==="
echo "Reuses the existing 41 precomputed pairs; no materialization or VAE preprocessing."
/workspace/wan_ticft/evaluation/check_gpu_free.sh "$GPU_ID" 42000

for path in "$ENV/bin/python" "$CONFIG" "$DATA/dataset.json"; do
  [[ -e "$path" ]] || { echo "Missing input: $path" >&2; exit 2; }
done

export PATH="$ENV/bin:$PATH"

echo "[1/4] Repair upstream absolute-path text embedding layout"
mkdir -p "$DATA/.precomputed/conditions"
while IFS= read -r latent; do
  rel="${latent#"$DATA/.precomputed/latents/"}"
  source_embedding="$DATA/$rel"
  target_embedding="$DATA/.precomputed/conditions/$rel"
  [[ -f "$source_embedding" ]] || {
    echo "Missing text embedding for $rel: $source_embedding" >&2
    exit 2
  }
  mkdir -p "$(dirname "$target_embedding")"
  cp -f "$source_embedding" "$target_embedding"
done < <(find "$DATA/.precomputed/latents" -type f -name '*.pt' | sort)

echo "[2/4] Switch the 2B smoke from Quanto FP8 to native BF16"
sed -i -E 's/^([[:space:]]*)quantization:.*$/\1quantization: null/' "$CONFIG"

latent_count="$(find "$DATA/.precomputed/latents" -type f -name '*.pt' | wc -l)"
condition_count="$(find "$DATA/.precomputed/conditions" -type f -name '*.pt' | wc -l)"
reference_count="$(find "$DATA/.precomputed/reference_latents" -type f -name '*.pt' | wc -l)"
echo "precomputed: latents=$latent_count conditions=$condition_count references=$reference_count"
[[ "$latent_count" -ge 32 && "$latent_count" -eq "$condition_count" && "$latent_count" -eq "$reference_count" ]] || {
  echo "Incomplete precomputed IC-LoRA dataset" >&2
  exit 2
}

echo "[3/4] Validate paired dataset before loading the model"
cd "$TRAINER"
"$ENV/bin/python" - <<PY
from ltxv_trainer.datasets import PrecomputedDataset

dataset = PrecomputedDataset(
    "$DATA/.precomputed",
    {
        "latents": "latents",
        "conditions": "conditions",
        "reference_latents": "ref_latents",
    },
)
assert len(dataset) == $latent_count, (len(dataset), $latent_count)
sample = dataset[0]
assert {"latents", "conditions", "ref_latents"} <= sample.keys()
print(f"paired_dataset={len(dataset)}")
PY

echo "[4/4] Train 200-step BF16 IC-LoRA smoke"
rm -rf "$TRAIN_OUTPUT"
CUDA_VISIBLE_DEVICES="$GPU_ID" HF_HOME=/root/.cache/huggingface \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
"$ENV/bin/python" scripts/train.py "$CONFIG"

test -f "$TRAIN_OUTPUT/checkpoints/lora_weights_step_00100.safetensors"
test -f "$TRAIN_OUTPUT/checkpoints/lora_weights_step_00200.safetensors"

echo
echo "Smoke complete. Review step0/100/200 validation videos at:"
echo "  $TRAIN_OUTPUT/samples"
echo "Checkpoints:"
echo "  $TRAIN_OUTPUT/checkpoints"

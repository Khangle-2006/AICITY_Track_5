#!/usr/bin/env bash
set -euo pipefail

cd /workspace

PYTHON=/opt/conda/envs/wan22/bin/python
ROOT=/workspace/experiments/wan_ticft
TRAINER=/workspace/third_party/LTX-Video-Trainer
ENV="$ROOT/ltxv_trainer_env"
DATA="$ROOT/ltx2b_ic_lora_wts_ip_data"
CONFIG="$DATA/ic_lora_2b_ip_smoke.yaml"
TRAIN_OUTPUT="$ROOT/ltx2b_ic_lora_wts_ip_smoke"
SOURCE_TRAIN="$ROOT/manifests/train_ticft_balanced_300_frames.jsonl"
SOURCE_VAL="$ROOT/manifests/val_ticft_balanced_125_frames.jsonl"
TRAIN_MANIFEST="$DATA/train_ip60_manifest.jsonl"
VAL_MANIFEST="$DATA/val_ip8_manifest.jsonl"
TRAIN_FRAMES="$DATA/train_manifest_frames"
VAL_FRAMES="$DATA/val_manifest_frames"
TRAIN_BASE="$DATA/train_region_base"
VAL_BASE="$DATA/val_region_base"
GPU_ID="${CUDA_VISIBLE_DEVICES:-3}"
GPU_ID="${GPU_ID%%,*}"

echo "=== LTX-Video 2B IC-LoRA supervised WTS IP-camera smoke ==="
echo "IP-camera train pool60; at least 32 paired clips; 8 independent validation references."
echo "512x288x9, rank16, 200 steps. This is not a submission builder."
/workspace/wan_ticft/evaluation/check_gpu_free.sh "$GPU_ID" 42000

for path in "$SOURCE_TRAIN" "$SOURCE_VAL"; do
  [[ -f "$path" ]] || { echo "Missing input: $path" >&2; exit 2; }
done

echo "[1/8] Install/resume official LTX-Video-Trainer"
if [[ ! -d "$TRAINER/.git" ]]; then
  git clone --depth 1 https://github.com/Lightricks/LTX-Video-Trainer "$TRAINER"
fi
if [[ ! -x "$ENV/bin/python" ]]; then
  /opt/conda/envs/wan22/bin/python -m venv "$ENV"
fi
if [[ ! -x "$ENV/bin/uv" ]]; then
  "$ENV/bin/pip" install --no-cache-dir uv
fi
UV_PROJECT_ENVIRONMENT="$ENV" "$ENV/bin/uv" sync --project "$TRAINER"
export PATH="$ENV/bin:$PATH"

echo "[2/8] Materialize test-like IP-camera train60 and val8 manifests"
"$PYTHON" -m wan_ticft.data.materialize_manifest_frames \
  --src_jsonl "$SOURCE_TRAIN" \
  --out_jsonl "$TRAIN_MANIFEST" \
  --frames_root "$TRAIN_FRAMES" \
  --view_type wts_vehicle_or_ip_camera \
  --source_contains normal_trimmed \
  --limit 60 \
  --clean
"$PYTHON" -m wan_ticft.data.materialize_manifest_frames \
  --src_jsonl "$SOURCE_VAL" \
  --out_jsonl "$VAL_MANIFEST" \
  --frames_root "$VAL_FRAMES" \
  --view_type wts_vehicle_or_ip_camera \
  --source_contains normal_trimmed \
  --limit 8 \
  --clean

echo "[3/8] Build correctly guarded region-medium/ground-truth train pairs"
"$PYTHON" -m wan_ticft.evaluation.materialize_wts_distribution_gate \
  --manifest "$TRAIN_MANIFEST" \
  --out_dir "$TRAIN_BASE" \
  --views wts_vehicle_or_ip_camera \
  --limit_per_domain 60 \
  --cover_manifest_targets \
  --candidates region_norm_medium \
  --clean

echo "[4/8] Build independent IP-camera validation references/targets"
"$PYTHON" -m wan_ticft.evaluation.materialize_wts_distribution_gate \
  --manifest "$VAL_MANIFEST" \
  --out_dir "$VAL_BASE" \
  --views wts_vehicle_or_ip_camera \
  --limit_per_domain 8 \
  --cover_manifest_targets \
  --candidates region_norm_medium \
  --clean

echo "[5/8] Build paired metadata and bounded smoke config"
rm -rf "$DATA/.precomputed"
"$PYTHON" -m wan_ticft.evaluation.prepare_ltx_ic_lora_dataset \
  --train_manifest "$TRAIN_MANIFEST" \
  --train_reference_report "$TRAIN_BASE/region_norm_medium_report.csv" \
  --train_target_report "$TRAIN_BASE/target_report.csv" \
  --eval_manifest "$VAL_MANIFEST" \
  --eval_reference_report "$VAL_BASE/region_norm_medium_report.csv" \
  --out_dir "$DATA" \
  --config_path "$CONFIG" \
  --training_output "$TRAIN_OUTPUT" \
  --frames 9 \
  --min_pairs 32 \
  --validation_count 2

echo "[6/8] Precompute target/reference latents and text embeddings"
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

# Upstream process_captions.py treats absolute media paths as absolute output
# paths. Move those embeddings into the relative tree expected by the trainer.
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

latent_count="$(find "$DATA/.precomputed/latents" -type f -name '*.pt' | wc -l)"
condition_count="$(find "$DATA/.precomputed/conditions" -type f -name '*.pt' | wc -l)"
reference_count="$(find "$DATA/.precomputed/reference_latents" -type f -name '*.pt' | wc -l)"
echo "precomputed: latents=$latent_count conditions=$condition_count references=$reference_count"
[[ "$latent_count" -ge 32 && "$latent_count" -eq "$condition_count" && "$latent_count" -eq "$reference_count" ]] || {
  echo "Incomplete precomputed IC-LoRA dataset" >&2
  exit 2
}

echo "[7/8] Train IC-LoRA smoke"
rm -rf "$TRAIN_OUTPUT"
CUDA_VISIBLE_DEVICES="$GPU_ID" HF_HOME=/root/.cache/huggingface \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
"$ENV/bin/python" scripts/train.py "$CONFIG"

echo "[8/8] Verify expected artifacts"
test -f "$TRAIN_OUTPUT/checkpoints/lora_weights_step_00100.safetensors"
test -f "$TRAIN_OUTPUT/checkpoints/lora_weights_step_00200.safetensors"
find "$TRAIN_OUTPUT" -maxdepth 3 -type f -printf '%p\n' | sort

echo
echo "Smoke complete. Review step0/100/200 validation videos at:"
echo "  $TRAIN_OUTPUT/samples"
echo "Ground-truth validation videos:"
echo "  $VAL_BASE/target"
echo "Checkpoints:"
echo "  $TRAIN_OUTPUT/checkpoints"
echo "Do not promote until samples are visually compared and a held-out metric gate is built."

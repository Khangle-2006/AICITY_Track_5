#!/usr/bin/env bash
set -euo pipefail
# IC-LoRA Refiner-A (BDD): reference = region_norm_medium warp (geometry-correct) -> target = GT.
# Mirrors run_ltx2b_ic_lora_wts_smoke.sh but view_type=bdd, candidate=region_norm_medium.
cd /workspace

PYTHON=/opt/conda/envs/wan22/bin/python
ROOT=/workspace/experiments/wan_ticft
TRAINER=/workspace/third_party/LTX-Video-Trainer
ENV="$ROOT/ltxv_trainer_env"
DATA="$ROOT/ltx2b_ic_lora_bdd_warp_data"
CONFIG="$DATA/ic_lora_2b_bdd_warp_smoke.yaml"
TRAIN_OUTPUT="$ROOT/ltx2b_ic_lora_bdd_warp_smoke"
SOURCE_TRAIN="$ROOT/manifests/train_ticft_balanced_300_frames.jsonl"
SOURCE_VAL="$ROOT/manifests/val_ticft_balanced_125_frames.jsonl"
TRAIN_MANIFEST="$DATA/train_bdd60_manifest.jsonl"
VAL_MANIFEST="$DATA/val_bdd8_manifest.jsonl"
TRAIN_FRAMES="$DATA/train_manifest_frames"
VAL_FRAMES="$DATA/val_manifest_frames"
TRAIN_BASE="$DATA/train_region_base"
VAL_BASE="$DATA/val_region_base"
TRAIN_LIMIT="${TRAIN_LIMIT:-60}"
VAL_LIMIT="${VAL_LIMIT:-8}"
GPU_ID="${CUDA_VISIBLE_DEVICES:-3}"
GPU_ID="${GPU_ID%%,*}"

echo "=== LTX-Video 2B IC-LoRA Refiner-A (BDD warp->GT) smoke ==="
echo "reference=region_norm_medium  target=GT  view=bdd  512x288x9 rank16 200 steps"
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

echo "[2/8] Materialize BDD train$TRAIN_LIMIT and val$VAL_LIMIT manifests"
"$PYTHON" -m wan_ticft.data.materialize_manifest_frames \
  --src_jsonl "$SOURCE_TRAIN" --out_jsonl "$TRAIN_MANIFEST" --frames_root "$TRAIN_FRAMES" \
  --view_type bdd --limit "$TRAIN_LIMIT" --clean
"$PYTHON" -m wan_ticft.data.materialize_manifest_frames \
  --src_jsonl "$SOURCE_VAL" --out_jsonl "$VAL_MANIFEST" --frames_root "$VAL_FRAMES" \
  --view_type bdd --limit "$VAL_LIMIT" --clean

echo "[3/8] Build region_norm_medium reference + GT target train pairs (view=bdd)"
"$PYTHON" -m wan_ticft.evaluation.materialize_wts_distribution_gate \
  --manifest "$TRAIN_MANIFEST" --out_dir "$TRAIN_BASE" \
  --views bdd --limit_per_domain "$TRAIN_LIMIT" --cover_manifest_targets \
  --candidates region_norm_medium --clean

echo "[4/8] Build independent BDD validation references/targets"
"$PYTHON" -m wan_ticft.evaluation.materialize_wts_distribution_gate \
  --manifest "$VAL_MANIFEST" --out_dir "$VAL_BASE" \
  --views bdd --limit_per_domain "$VAL_LIMIT" --cover_manifest_targets \
  --candidates region_norm_medium --clean

echo "[5/8] Build paired metadata and bounded smoke config"
rm -rf "$DATA/.precomputed"
"$PYTHON" -m wan_ticft.evaluation.prepare_ltx_ic_lora_dataset \
  --train_manifest "$TRAIN_MANIFEST" \
  --train_reference_report "$TRAIN_BASE/region_norm_medium_report.csv" \
  --train_target_report "$TRAIN_BASE/target_report.csv" \
  --eval_manifest "$VAL_MANIFEST" \
  --eval_reference_report "$VAL_BASE/region_norm_medium_report.csv" \
  --out_dir "$DATA" --config_path "$CONFIG" --training_output "$TRAIN_OUTPUT" \
  --frames 9 --min_pairs 32 --validation_count 2 --steps 2000 --checkpoint_interval 500

echo "[6/8] Precompute target/reference latents and text embeddings"
cd "$TRAINER"
CUDA_VISIBLE_DEVICES="$GPU_ID" HF_HOME=/root/.cache/huggingface \
"$ENV/bin/python" scripts/preprocess_dataset.py "$DATA/dataset.json" \
  --resolution-buckets 512x288x9 --caption-column caption \
  --video-column media_path --reference-column reference_path \
  --model-source LTXV_2B_0.9.6_DEV --output-dir "$DATA/.precomputed" --batch-size 1
while IFS= read -r latent; do
  rel="${latent#"$DATA/.precomputed/latents/"}"
  src_emb="$DATA/$rel"; tgt_emb="$DATA/.precomputed/conditions/$rel"
  [[ -f "$tgt_emb" ]] && continue   # preprocess already placed conditions correctly
  [[ -f "$src_emb" ]] || continue   # nothing to relocate
  mkdir -p "$(dirname "$tgt_emb")"; cp -f "$src_emb" "$tgt_emb"
done < <(find "$DATA/.precomputed/latents" -type f -name '*.pt' | sort)
lc="$(find "$DATA/.precomputed/latents" -type f -name '*.pt' | wc -l)"
cc="$(find "$DATA/.precomputed/conditions" -type f -name '*.pt' | wc -l)"
rc="$(find "$DATA/.precomputed/reference_latents" -type f -name '*.pt' | wc -l)"
echo "precomputed: latents=$lc conditions=$cc references=$rc"
[[ "$lc" -ge 32 && "$lc" -eq "$cc" && "$lc" -eq "$rc" ]] || { echo "Incomplete precomputed dataset" >&2; exit 2; }

echo "[7/8] Train IC-LoRA smoke"
rm -rf "$TRAIN_OUTPUT"
CUDA_VISIBLE_DEVICES="$GPU_ID" HF_HOME=/root/.cache/huggingface \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
"$ENV/bin/python" scripts/train.py "$CONFIG"

echo "[8/8] Verify checkpoints"
test -f "$TRAIN_OUTPUT/checkpoints/lora_weights_step_00100.safetensors"
test -f "$TRAIN_OUTPUT/checkpoints/lora_weights_step_00200.safetensors"
echo
echo "Refiner-A smoke done. Samples: $TRAIN_OUTPUT/samples | GT: $VAL_BASE/target | ckpts: $TRAIN_OUTPUT/checkpoints"

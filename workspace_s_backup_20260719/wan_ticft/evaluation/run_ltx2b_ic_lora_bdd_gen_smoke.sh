#!/bin/bash
set -euo pipefail
# IC-LoRA Refiner-B (BDD): reference = ltx2b CHUNKED generation -> target = GT.
# Reordered: gen chunk refs from a LARGE pool (train videos decode unreliably ~23% pass),
# keep REF_TRAIN good ones, then materialize GT only for the passing samples.
cd /workspace
PYTHON=${PYTHON:-/opt/conda/envs/wan22/bin/python}
ROOT=/workspace/experiments/wan_ticft
TRAINER=/workspace/third_party/LTX-Video-Trainer
ENV="$ROOT/ltxv_trainer_env"
LTX_REPO=/workspace/third_party/LTX-Video
LTX_PYTHON="${LTX_PYTHON:-$ROOT/ltxv_env/bin/python}"
PIPELINE_CONFIG="${PIPELINE_CONFIG:-/workspace/wan_ticft/evaluation/ltxv_2b_gate.yaml}"
SOURCE_TRAIN="$ROOT/manifests/train_ticft_balanced_300_frames.jsonl"
SOURCE_VAL="$ROOT/manifests/val_ticft_balanced_125_frames.jsonl"
DATA="$ROOT/ltx2b_ic_lora_bdd_gen_data"
CONFIG="$DATA/ic_lora_2b_bdd_gen_smoke.yaml"
TRAIN_OUTPUT="$ROOT/ltx2b_ic_lora_bdd_gen_smoke"
TRAIN_MANIFEST="$DATA/train_pool_manifest.jsonl"
VAL_MANIFEST="$DATA/val_pool_manifest.jsonl"
TRAIN_FRAMES="$DATA/train_manifest_frames"; VAL_FRAMES="$DATA/val_manifest_frames"
TRAIN_GT="$DATA/train_gt_base"; VAL_GT="$DATA/val_gt_base"
TRAIN_REF="$DATA/train_ltx2b_chunked_ref"; VAL_REF="$DATA/val_ltx2b_chunked_ref"
TRAIN_FILT="$DATA/train_filtered_manifest.jsonl"; VAL_FILT="$DATA/val_filtered_manifest.jsonl"
TRAIN_POOL="${TRAIN_POOL:-220}"; REF_TRAIN="${REF_TRAIN:-36}"
VAL_POOL="${VAL_POOL:-24}"; REF_VAL="${REF_VAL:-6}"; MIN_PAIRS="${MIN_PAIRS:-32}"
CHUNK_FRAMES="${CHUNK_FRAMES:-24}"; HEIGHT="${HEIGHT:-288}"; WIDTH="${WIDTH:-512}"
GPU_ID="${CUDA_VISIBLE_DEVICES:-3}"; GPU_ID="${GPU_ID%%,*}"
export CUDA_VISIBLE_DEVICES="$GPU_ID"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

echo "=== IC-LoRA Refiner-B (chunk->GT) | pool train=$TRAIN_POOL ref_target=$REF_TRAIN ==="
/workspace/wan_ticft/evaluation/check_gpu_free.sh "$GPU_ID" 42000
for p in "$SOURCE_TRAIN" "$SOURCE_VAL" "$PIPELINE_CONFIG"; do [[ -f "$p" ]] || { echo "Missing $p" >&2; exit 2; }; done

echo "[1/9] Trainer env"
if [[ ! -d "$TRAINER/.git" ]]; then git clone --depth 1 https://github.com/Lightricks/LTX-Video-Trainer "$TRAINER"; fi
if [[ ! -x "$ENV/bin/python" ]]; then /opt/conda/envs/wan22/bin/python -m venv "$ENV"; fi
if [[ ! -x "$ENV/bin/uv" ]]; then "$ENV/bin/pip" install --no-cache-dir uv; fi
UV_PROJECT_ENVIRONMENT="$ENV" "$ENV/bin/uv" sync --project "$TRAINER"

echo "[2/9] Materialize LARGE train/val pools"
"$PYTHON" -m wan_ticft.data.materialize_manifest_frames --src_jsonl "$SOURCE_TRAIN" --out_jsonl "$TRAIN_MANIFEST" --frames_root "$TRAIN_FRAMES" --view_type bdd --limit "$TRAIN_POOL" --clean
"$PYTHON" -m wan_ticft.data.materialize_manifest_frames --src_jsonl "$SOURCE_VAL" --out_jsonl "$VAL_MANIFEST" --frames_root "$VAL_FRAMES" --view_type bdd --limit "$VAL_POOL" --clean

echo "[3/9] Generate ltx2b CHUNKED refs (keep REF_TRAIN good from pool)"
"$PYTHON" -m wan_ticft.evaluation.run_ltxv_val_chunked_gate --manifest "$TRAIN_MANIFEST" --out_dir "$TRAIN_REF" \
  --ltx_repo "$LTX_REPO" --ltx_python "$LTX_PYTHON" --pipeline_config "$PIPELINE_CONFIG" \
  --limit "$REF_TRAIN" --view_type bdd --prompt_mode traffic_detail --height "$HEIGHT" --width "$WIDTH" --chunk_frames "$CHUNK_FRAMES" --skip_bad
"$PYTHON" -m wan_ticft.evaluation.run_ltxv_val_chunked_gate --manifest "$VAL_MANIFEST" --out_dir "$VAL_REF" \
  --ltx_repo "$LTX_REPO" --ltx_python "$LTX_PYTHON" --pipeline_config "$PIPELINE_CONFIG" \
  --limit "$REF_VAL" --view_type bdd --prompt_mode traffic_detail --height "$HEIGHT" --width "$WIDTH" --chunk_frames "$CHUNK_FRAMES" --skip_bad

echo "[4/9] Filter manifests to samples that produced chunk refs"
for pair in "$TRAIN_REF $TRAIN_MANIFEST $TRAIN_FILT" "$VAL_REF $VAL_MANIFEST $VAL_FILT"; do
  set -- $pair
  "$PYTHON" - "$1/report.csv" "$2" "$3" <<'PY'
import csv,json,sys
ids={r["sample_id"] for r in csv.DictReader(open(sys.argv[1],encoding="utf-8")) if r.get("status")=="ok"}
out=[l for l in open(sys.argv[2],encoding="utf-8") if l.strip() and json.loads(l)["sample_id"] in ids]
open(sys.argv[3],"w",encoding="utf-8").writelines(out)
print(f"filtered {sys.argv[3]}: {len(out)} rows")
PY
done

echo "[5/9] GT target videos ONLY for passing samples"
"$PYTHON" -m wan_ticft.evaluation.materialize_wts_distribution_gate --manifest "$TRAIN_FILT" --out_dir "$TRAIN_GT" --views bdd --limit_per_domain "$REF_TRAIN" --cover_manifest_targets --candidates region_norm_medium --clean
"$PYTHON" -m wan_ticft.evaluation.materialize_wts_distribution_gate --manifest "$VAL_FILT" --out_dir "$VAL_GT" --views bdd --limit_per_domain "$REF_VAL" --cover_manifest_targets --candidates region_norm_medium --clean

echo "[6/9] Build paired metadata + config"
rm -rf "$DATA/.precomputed"
"$PYTHON" -m wan_ticft.evaluation.prepare_ltx_ic_lora_dataset --train_manifest "$TRAIN_FILT" \
  --train_reference_report "$TRAIN_REF/report.csv" --train_target_report "$TRAIN_GT/target_report.csv" \
  --eval_manifest "$VAL_FILT" --eval_reference_report "$VAL_REF/report.csv" \
  --out_dir "$DATA" --config_path "$CONFIG" --training_output "$TRAIN_OUTPUT" --frames 9 --min_pairs "$MIN_PAIRS" --validation_count 2 --steps 2000 --checkpoint_interval 500

echo "[7/9] Precompute latents"
cd "$TRAINER"
CUDA_VISIBLE_DEVICES="$GPU_ID" HF_HOME=/root/.cache/huggingface "$ENV/bin/python" scripts/preprocess_dataset.py "$DATA/dataset.json" \
  --resolution-buckets 512x288x9 --caption-column caption --video-column media_path --reference-column reference_path \
  --model-source LTXV_2B_0.9.6_DEV --output-dir "$DATA/.precomputed" --batch-size 1
while IFS= read -r latent; do
  rel="${latent#"$DATA/.precomputed/latents/"}"; src_emb="$DATA/$rel"; tgt_emb="$DATA/.precomputed/conditions/$rel"
  [[ -f "$tgt_emb" ]] && continue
  [[ -f "$src_emb" ]] || continue
  mkdir -p "$(dirname "$tgt_emb")"; cp -f "$src_emb" "$tgt_emb"
done < <(find "$DATA/.precomputed/latents" -type f -name '*.pt' | sort)
lc="$(find "$DATA/.precomputed/latents" -type f -name '*.pt' | wc -l)"; rc="$(find "$DATA/.precomputed/reference_latents" -type f -name '*.pt' | wc -l)"
echo "precomputed latents=$lc references=$rc"
[[ "$lc" -ge "$MIN_PAIRS" && "$lc" -eq "$rc" ]] || { echo "Incomplete precomputed dataset (latents=$lc refs=$rc)" >&2; exit 2; }

echo "[8/9] Train"
rm -rf "$TRAIN_OUTPUT"
CUDA_VISIBLE_DEVICES="$GPU_ID" HF_HOME=/root/.cache/huggingface PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True "$ENV/bin/python" scripts/train.py "$CONFIG"

echo "[9/9] Done. ckpt: $TRAIN_OUTPUT/checkpoints"

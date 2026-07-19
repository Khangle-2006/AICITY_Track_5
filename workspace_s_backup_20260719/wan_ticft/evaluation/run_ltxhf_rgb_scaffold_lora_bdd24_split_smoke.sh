#!/usr/bin/env bash
set -euo pipefail

cd /workspace

ROOT=/workspace/experiments/wan_ticft
PYTHON=/opt/conda/envs/wan22/bin/python
TRAINER=/workspace/third_party/LTX-Video-Trainer
ENV="$ROOT/ltxv_trainer_env"
RUN="$ROOT/ltx2b_ltxhf_rgb_scaffold_lora_bdd24split_train20_s100"
DATA="$RUN/ic_lora_data"
CONFIG="$DATA/ic_lora_2b_ltxhf_rgb_scaffold_bdd24split.yaml"
TRAIN_OUTPUT="$RUN/train_output"
SCORE_OUT="$RUN/validation_score"

SOURCE_MANIFEST="$ROOT/ltxv_bdd25_gate/selected_manifest.jsonl"
SCAFFOLD_REPORT="$ROOT/ltx_detail_donor_bdd24_gate/ltxhf_t55/report.csv"
TARGET_REPORT="$ROOT/ltx2b_depth_scaffold_lora_bdd_train48_s200/eval_target/target_report.csv"
TRAIN_MANIFEST="$RUN/train20_manifest.jsonl"
VAL_MANIFEST="$RUN/val4_manifest.jsonl"

GPU_ID="${CUDA_VISIBLE_DEVICES:-0}"
GPU_ID="${GPU_ID%%,*}"
FRAMES="${FRAMES:-9}"
WIDTH="${WIDTH:-512}"
HEIGHT="${HEIGHT:-288}"
TRAIN_STEPS="${TRAIN_STEPS:-100}"
CHECKPOINT_INTERVAL="${CHECKPOINT_INTERVAL:-50}"
RANK="${RANK:-16}"
SCORE_STEPS="${SCORE_STEPS:-0 50 100}"

echo "=== LTX2B RGB ltxhf scaffold -> real LoRA BDD24 split smoke ==="
echo "run=$RUN"
echo "gpu=$GPU_ID frames=$FRAMES dims=${WIDTH}x${HEIGHT} steps=$TRAIN_STEPS"

for path in "$SOURCE_MANIFEST" "$SCAFFOLD_REPORT" "$TARGET_REPORT"; do
  [[ -s "$path" ]] || { echo "Missing input: $path" >&2; exit 2; }
done
[[ -x "$ENV/bin/python" ]] || { echo "Missing trainer env: $ENV" >&2; exit 2; }

mkdir -p "$RUN"
"$PYTHON" - <<PY
from pathlib import Path
import json
src = Path("$SOURCE_MANIFEST")
train = Path("$TRAIN_MANIFEST")
val = Path("$VAL_MANIFEST")
rows = [json.loads(line) for line in src.read_text(encoding="utf-8").splitlines() if line.strip() and json.loads(line).get("view_type") == "bdd"]
if len(rows) < 24:
    raise SystemExit(f"need 24 BDD rows, got {len(rows)}")
train.parent.mkdir(parents=True, exist_ok=True)
train.write_text("".join(json.dumps(row) + "\\n" for row in rows[:20]), encoding="utf-8")
val.write_text("".join(json.dumps(row) + "\\n" for row in rows[20:24]), encoding="utf-8")
print(f"train={train} rows=20")
print(f"val={val} rows=4")
PY

echo "[1/5] Build paired metadata/config"
rm -rf "$DATA/.precomputed" "$TRAIN_OUTPUT" "$SCORE_OUT"
"$PYTHON" -m wan_ticft.evaluation.prepare_ltx_ic_lora_dataset \
  --train_manifest "$TRAIN_MANIFEST" \
  --train_reference_report "$SCAFFOLD_REPORT" \
  --train_target_report "$TARGET_REPORT" \
  --eval_manifest "$VAL_MANIFEST" \
  --eval_reference_report "$SCAFFOLD_REPORT" \
  --out_dir "$DATA" \
  --config_path "$CONFIG" \
  --training_output "$TRAIN_OUTPUT" \
  --frames "$FRAMES" \
  --min_pairs 20 \
  --validation_count 4 \
  --prompt "REALISTIC_TRAFFIC_REFINER Convert this ltxhf RGB traffic scaffold into real dashcam future video. Preserve exact road geometry, camera motion, actor identity, and object positions. Remove ghosting, synthetic smear, and afterimages." \
  --steps "$TRAIN_STEPS" \
  --rank "$RANK" \
  --width "$WIDTH" \
  --height "$HEIGHT" \
  --checkpoint_interval "$CHECKPOINT_INTERVAL"

echo "[2/5] Precompute target/reference latents"
cd "$TRAINER"
CUDA_VISIBLE_DEVICES="$GPU_ID" HF_HOME=/root/.cache/huggingface \
"$ENV/bin/python" scripts/preprocess_dataset.py "$DATA/dataset.json" \
  --resolution-buckets "${WIDTH}x${HEIGHT}x${FRAMES}" \
  --caption-column caption \
  --video-column media_path \
  --reference-column reference_path \
  --model-source LTXV_2B_0.9.6_DEV \
  --output-dir "$DATA/.precomputed" \
  --batch-size 1

while IFS= read -r latent; do
  rel="${latent#"$DATA/.precomputed/latents/"}"
  target_embedding="$DATA/.precomputed/conditions/$rel"
  if [[ ! -f "$target_embedding" ]]; then
    source_embedding="$DATA/$rel"
    [[ -f "$source_embedding" ]] || { echo "Missing text embedding for $rel" >&2; exit 2; }
    mkdir -p "$(dirname "$target_embedding")"
    cp -f "$source_embedding" "$target_embedding"
  fi
done < <(find "$DATA/.precomputed/latents" -type f -name '*.pt' | sort)

latent_count="$(find "$DATA/.precomputed/latents" -type f -name '*.pt' | wc -l)"
condition_count="$(find "$DATA/.precomputed/conditions" -type f -name '*.pt' | wc -l)"
reference_count="$(find "$DATA/.precomputed/reference_latents" -type f -name '*.pt' | wc -l)"
echo "precomputed: latents=$latent_count conditions=$condition_count references=$reference_count"
[[ "$latent_count" -eq 20 && "$condition_count" -eq 20 && "$reference_count" -eq 20 ]] || {
  echo "Incomplete precomputed IC-LoRA dataset" >&2
  exit 2
}

echo "[3/5] Train LoRA"
CUDA_VISIBLE_DEVICES="$GPU_ID" HF_HOME=/root/.cache/huggingface \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
"$ENV/bin/python" scripts/train.py "$CONFIG"

echo "[4/5] Score validation samples"
cd /workspace
CUDA_VISIBLE_DEVICES="$GPU_ID" TORCH_HOME="$ROOT/tools/torch_cache" \
"$PYTHON" -m wan_ticft.evaluation.score_ltx_scaffold_lora_validation \
  --validation_json "$DATA/validation.json" \
  --target_report "$TARGET_REPORT" \
  --scaffold_report "$SCAFFOLD_REPORT" \
  --samples "$TRAIN_OUTPUT/samples" \
  --out_dir "$SCORE_OUT" \
  --frames "$FRAMES" \
  --steps $SCORE_STEPS

echo "[5/5] Done"
echo "config=$CONFIG"
echo "samples=$TRAIN_OUTPUT/samples"
echo "checkpoints=$TRAIN_OUTPUT/checkpoints"
echo "score=$SCORE_OUT"

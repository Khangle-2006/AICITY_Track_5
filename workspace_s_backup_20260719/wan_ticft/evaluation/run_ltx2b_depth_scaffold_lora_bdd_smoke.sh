#!/usr/bin/env bash
set -euo pipefail

cd /workspace

PYTHON=/opt/conda/envs/wan22/bin/python
ROOT=/workspace/experiments/wan_ticft
TRAINER=/workspace/third_party/LTX-Video-Trainer
ENV="$ROOT/ltxv_trainer_env"
GPU_ID="${CUDA_VISIBLE_DEVICES:-2}"
GPU_ID="${GPU_ID%%,*}"

TRAIN_LIMIT="${TRAIN_LIMIT:-48}"
VAL_COUNT="${VAL_COUNT:-4}"
FRAMES="${FRAMES:-9}"
WIDTH="${WIDTH:-512}"
HEIGHT="${HEIGHT:-288}"
TRAIN_STEPS="${TRAIN_STEPS:-200}"
CHECKPOINT_INTERVAL="${CHECKPOINT_INTERVAL:-100}"
RANK="${RANK:-16}"
SCORE_STEPS="${SCORE_STEPS:-0 $CHECKPOINT_INTERVAL $TRAIN_STEPS}"

TRAIN_SOURCE="${TRAIN_SOURCE:-$ROOT/manifests/train_ticft_balanced_300_frames.jsonl}"
EVAL_BASE="${EVAL_BASE:-$ROOT/ltxv_bdd25_gate}"
MODEL_CACHE="$ROOT/depth_layered_models"
LTX_REPO=/workspace/third_party/LTX-Video
LTX_PY=/workspace/experiments/wan_ticft/ltxv_env/bin/python
LTX_CFG=/workspace/wan_ticft/evaluation/ltxv_2b_gate.yaml

RUN_ROOT="$ROOT/ltx2b_depth_scaffold_lora_bdd_train${TRAIN_LIMIT}_s${TRAIN_STEPS}"
LEGACY_TRAIN_ROOT="$ROOT/bdd_supervised_ltx_depth_train${TRAIN_LIMIT}"
if [[ -s "$LEGACY_TRAIN_ROOT/ltx_base/report.csv" && -s "$LEGACY_TRAIN_ROOT/depth/depth_layered_a75/report.csv" ]]; then
  TRAIN_BASE="$LEGACY_TRAIN_ROOT/ltx_base"
  TRAIN_DEPTH="$LEGACY_TRAIN_ROOT/depth"
else
  TRAIN_BASE="$RUN_ROOT/train_ltx_base"
  TRAIN_DEPTH="$RUN_ROOT/train_depth"
fi
TRAIN_REPAIRED_MANIFEST="$TRAIN_BASE/selected_manifest_repaired_conditions.jsonl"
TRAIN_REPAIRED_FRAMES="$TRAIN_BASE/repaired_condition_frames"
TRAIN_TARGET="$RUN_ROOT/train_target"
EVAL_DEPTH="$ROOT/bdd_supervised_ltx_depth_eval24/depth"
EVAL_TARGET="$RUN_ROOT/eval_target"
DATA="$RUN_ROOT/ic_lora_data"
CONFIG="$DATA/ic_lora_2b_depth_scaffold_bdd.yaml"
TRAIN_OUTPUT="$RUN_ROOT/train_output"
SCORE_OUT="$RUN_ROOT/validation_score"

echo "=== LTX2B depth_a75 scaffold-conditioned LoRA BDD smoke ==="
echo "Goal: learn depth_a75 scaffold -> GT target, not full future generation from scratch."
echo "train_limit=$TRAIN_LIMIT val_count=$VAL_COUNT frames=$FRAMES dims=${WIDTH}x${HEIGHT} steps=$TRAIN_STEPS rank=$RANK"
/workspace/wan_ticft/evaluation/check_gpu_free.sh "$GPU_ID" 42000

for path in "$TRAIN_SOURCE" "$EVAL_BASE/selected_manifest.jsonl" "$EVAL_BASE/report.csv"; do
  [[ -s "$path" ]] || { echo "Missing input: $path" >&2; exit 2; }
done

echo "[1/9] Install/resume official LTX-Video-Trainer"
if [[ ! -d "$TRAINER/.git" ]]; then
  git clone --depth 1 https://github.com/Lightricks/LTX-Video-Trainer "$TRAINER"
fi
if [[ ! -x "$ENV/bin/python" ]]; then
  "$PYTHON" -m venv "$ENV"
fi
if [[ ! -x "$ENV/bin/uv" ]]; then
  "$ENV/bin/pip" install --no-cache-dir uv
fi
UV_PROJECT_ENVIRONMENT="$ENV" "$ENV/bin/uv" sync --project "$TRAINER"
export PATH="$ENV/bin:$PATH"

echo "[2/9] Generate/resume BDD train LTX base"
if [[ -s "$TRAIN_BASE/selected_manifest.jsonl" && -s "$TRAIN_BASE/report.csv" ]]; then
  echo "  found existing train base; skip"
else
  CUDA_VISIBLE_DEVICES="$GPU_ID" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  "$PYTHON" -m wan_ticft.evaluation.run_ltxv_val_gate \
    --manifest "$TRAIN_SOURCE" \
    --out_dir "$TRAIN_BASE" \
    --ltx_repo "$LTX_REPO" \
    --ltx_python "$LTX_PY" \
    --pipeline_config "$LTX_CFG" \
    --limit "$TRAIN_LIMIT" \
    --view_type bdd \
    --max_history_frames 41 \
    --prompt_mode behavior \
    --skip_bad \
    --resume
fi

echo "[3/9] Repair train manifest condition frames from LTX history videos"
"$PYTHON" -m wan_ticft.evaluation.repair_ltx_selected_manifest_conditions \
  --manifest "$TRAIN_BASE/selected_manifest.jsonl" \
  --history_dir "$TRAIN_BASE/history_videos" \
  --frames_root "$TRAIN_REPAIRED_FRAMES" \
  --out_manifest "$TRAIN_REPAIRED_MANIFEST" \
  --skip_bad \
  --clean

echo "[4/9] Generate/resume train depth_a75 scaffold over LTX base"
if [[ -s "$TRAIN_DEPTH/depth_layered_a75/report.csv" ]]; then
  echo "  found existing train depth scaffold; skip"
else
  CUDA_VISIBLE_DEVICES="$GPU_ID" HF_HOME="$MODEL_CACHE/hf_home" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  "$PYTHON" -m wan_ticft.evaluation.materialize_depth_layered_renderer \
    --manifest "$TRAIN_REPAIRED_MANIFEST" \
    --base_report "$TRAIN_BASE/report.csv" \
    --out_root "$TRAIN_DEPTH" \
    --model_cache "$MODEL_CACHE/huggingface" \
    --view_type bdd \
    --preserve_full_video \
    --limit "$TRAIN_LIMIT" \
    --alphas 0.75 \
    --focal_scale 0.9 \
    --point_stride 2 \
    --splat_radius 1 \
    --min_matches 20 \
    --min_inliers 12 \
    --photo_sigma 32 \
    --segment_batch 4 \
    --pose_smoothing_window 7 \
    --filled_pose_scale 0.35 \
    --confidence_ema 0.65
fi

echo "[5/9] Generate/resume eval depth_a75 scaffold and target reports"
if [[ -s "$EVAL_DEPTH/depth_layered_a75/report.csv" ]]; then
  echo "  found existing BDD24 eval depth scaffold; skip"
else
  CUDA_VISIBLE_DEVICES="$GPU_ID" HF_HOME="$MODEL_CACHE/hf_home" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  "$PYTHON" -m wan_ticft.evaluation.materialize_depth_layered_renderer \
    --manifest "$EVAL_BASE/selected_manifest.jsonl" \
    --base_report "$EVAL_BASE/report.csv" \
    --out_root "$EVAL_DEPTH" \
    --model_cache "$MODEL_CACHE/huggingface" \
    --view_type bdd \
    --preserve_full_video \
    --limit 24 \
    --alphas 0.75 \
    --focal_scale 0.9 \
    --point_stride 2 \
    --splat_radius 1 \
    --min_matches 20 \
    --min_inliers 12 \
    --photo_sigma 32 \
    --segment_batch 4 \
    --pose_smoothing_window 7 \
    --filled_pose_scale 0.35 \
    --confidence_ema 0.65
fi
"$PYTHON" -m wan_ticft.evaluation.materialize_manifest_target_report \
  --manifest "$TRAIN_REPAIRED_MANIFEST" \
  --out_dir "$TRAIN_TARGET" \
  --limit "$TRAIN_LIMIT" \
  --view_type bdd \
  --prepend_guard \
  --clean
"$PYTHON" -m wan_ticft.evaluation.materialize_manifest_target_report \
  --manifest "$EVAL_BASE/selected_manifest.jsonl" \
  --out_dir "$EVAL_TARGET" \
  --limit 24 \
  --view_type bdd \
  --prepend_guard \
  --clean

echo "[6/9] Build paired metadata and LTX LoRA config"
rm -rf "$DATA/.precomputed"
"$PYTHON" -m wan_ticft.evaluation.prepare_ltx_ic_lora_dataset \
  --train_manifest "$TRAIN_REPAIRED_MANIFEST" \
  --train_reference_report "$TRAIN_DEPTH/depth_layered_a75/report.csv" \
  --train_target_report "$TRAIN_TARGET/target_report.csv" \
  --eval_manifest "$EVAL_BASE/selected_manifest.jsonl" \
  --eval_reference_report "$EVAL_DEPTH/depth_layered_a75/report.csv" \
  --out_dir "$DATA" \
  --config_path "$CONFIG" \
  --training_output "$TRAIN_OUTPUT" \
  --frames "$FRAMES" \
  --min_pairs "$(( TRAIN_LIMIT < 24 ? TRAIN_LIMIT : 24 ))" \
  --validation_count "$VAL_COUNT" \
  --prompt "Enhance this traffic forecast scaffold into realistic future video while preserving exact road geometry, camera motion, and actor positions." \
  --steps "$TRAIN_STEPS" \
  --rank "$RANK" \
  --width "$WIDTH" \
  --height "$HEIGHT" \
  --checkpoint_interval "$CHECKPOINT_INTERVAL"

echo "[7/9] Precompute target/reference latents and text embeddings"
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
    [[ -f "$source_embedding" ]] || {
      echo "Missing text embedding for $rel: $source_embedding" >&2
      exit 2
    }
    mkdir -p "$(dirname "$target_embedding")"
    cp -f "$source_embedding" "$target_embedding"
  fi
done < <(find "$DATA/.precomputed/latents" -type f -name '*.pt' | sort)

latent_count="$(find "$DATA/.precomputed/latents" -type f -name '*.pt' | wc -l)"
condition_count="$(find "$DATA/.precomputed/conditions" -type f -name '*.pt' | wc -l)"
reference_count="$(find "$DATA/.precomputed/reference_latents" -type f -name '*.pt' | wc -l)"
echo "precomputed: latents=$latent_count conditions=$condition_count references=$reference_count"
[[ "$latent_count" -ge 24 && "$latent_count" -eq "$condition_count" && "$latent_count" -eq "$reference_count" ]] || {
  echo "Incomplete precomputed IC-LoRA dataset" >&2
  exit 2
}

echo "[8/9] Train scaffold-conditioned LoRA"
rm -rf "$TRAIN_OUTPUT"
CUDA_VISIBLE_DEVICES="$GPU_ID" HF_HOME=/root/.cache/huggingface \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
"$ENV/bin/python" scripts/train.py "$CONFIG"

echo "[9/9] Score validation samples"
for step in $SCORE_STEPS; do
  if [[ "$step" != "0" ]]; then
    padded="$(printf "%05d" "$step")"
    test -f "$TRAIN_OUTPUT/checkpoints/lora_weights_step_${padded}.safetensors"
  fi
done
cd /workspace
CUDA_VISIBLE_DEVICES="$GPU_ID" TORCH_HOME="$ROOT/tools/torch_cache" \
"$PYTHON" -m wan_ticft.evaluation.score_ltx_scaffold_lora_validation \
  --validation_json "$DATA/validation.json" \
  --target_report "$EVAL_TARGET/target_report.csv" \
  --scaffold_report "$EVAL_DEPTH/depth_layered_a75/report.csv" \
  --samples "$TRAIN_OUTPUT/samples" \
  --out_dir "$SCORE_OUT" \
  --frames "$FRAMES" \
  --steps $SCORE_STEPS

echo
echo "Smoke complete."
echo "config=$CONFIG"
echo "samples=$TRAIN_OUTPUT/samples"
echo "checkpoints=$TRAIN_OUTPUT/checkpoints"
echo "score=$SCORE_OUT"
echo
echo "Gate:"
echo "  Continue only if step100/step200 beats scaffold on FID/FVD proxy or visual LPIPS-like quality,"
echo "  while PSNR/SSIM stay close and road/actor geometry does not hallucinate."

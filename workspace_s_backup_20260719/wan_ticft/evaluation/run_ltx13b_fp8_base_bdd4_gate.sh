#!/usr/bin/env bash
set -euo pipefail

cd /workspace

PYTHON=/opt/conda/envs/wan22/bin/python
HF=/opt/conda/envs/wan22/bin/hf
ROOT=/workspace/experiments/wan_ticft
MANIFEST="$ROOT/ltxv_bdd25_gate/dense40_manifest.jsonl"
EVAL_DEPTH="$ROOT/bdd_supervised_ltx_depth_eval24/depth"
EVAL_BASE="$ROOT/ltxv_bdd25_gate"
OUT="$ROOT/ltx13b_fp8_base_bdd4_gate"
MODEL_DIR=/workspace/models/ltx13b098_ic_detailer
MODEL="$MODEL_DIR/ltxv-13b-0.9.8-dev-fp8.safetensors"
UPSCALER="$MODEL_DIR/ltxv-spatial-upscaler-0.9.8.safetensors"
LTX_REPO=/workspace/third_party/LTX-Video
LTX_PY=/workspace/experiments/wan_ticft/ltxv_env/bin/python
LTX_CFG=/workspace/wan_ticft/evaluation/ltxv_13b_dev_fp8_local_gate.yaml
GPU_ID="${CUDA_VISIBLE_DEVICES:-0}"
GPU_ID="${GPU_ID%%,*}"
LIMIT="${LIMIT:-1}"
HEIGHT="${HEIGHT:-704}"
WIDTH="${WIDTH:-1216}"
NUM_FRAMES="${NUM_FRAMES:-121}"
SAMPLE_IDS=(
  val_bdd_pc_5k_video7040_phase0
  val_bdd_pc_5k_video8125_phase1
  val_bdd_pc_5k_video8139_phase2
  val_bdd_pc_5k_video9450_phase4
)

echo "=== LTX 13B FP8 base predictor BDD gate ==="
echo "Default LIMIT=1 full-res. Use LIMIT=4 only after the first sample passes visual/proxy gate."
echo "Resolution=${HEIGHT}x${WIDTH} frames=${NUM_FRAMES}"
echo "out=$OUT"
/workspace/wan_ticft/evaluation/check_gpu_free.sh "$GPU_ID" 26000

for path in "$MANIFEST" "$EVAL_BASE/report.csv" "$EVAL_DEPTH/depth_layered_a75/report.csv" "$LTX_CFG"; do
  [[ -s "$path" ]] || { echo "Missing input: $path" >&2; exit 2; }
done
mkdir -p "$MODEL_DIR"
if [[ ! -s "$MODEL" ]]; then
  echo "[download] LTX 13B 0.9.8 DEV FP8"
  "$HF" download Lightricks/LTX-Video ltxv-13b-0.9.8-dev-fp8.safetensors --local-dir "$MODEL_DIR"
fi
if [[ ! -s "$UPSCALER" ]]; then
  echo "[download] LTX spatial upscaler 0.9.8"
  "$HF" download Lightricks/LTX-Video ltxv-spatial-upscaler-0.9.8.safetensors --local-dir "$MODEL_DIR"
fi

echo "[1/4] Generate/resume LTX 13B FP8 base predictions"
CUDA_VISIBLE_DEVICES="$GPU_ID" \
HF_HOME=/root/.cache/huggingface \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
TOKENIZERS_PARALLELISM=false \
"$PYTHON" -m wan_ticft.evaluation.run_ltxv_val_gate \
  --manifest "$MANIFEST" \
  --out_dir "$OUT" \
  --ltx_repo "$LTX_REPO" \
  --ltx_python "$LTX_PY" \
  --pipeline_config "$LTX_CFG" \
  --limit "$LIMIT" \
  --view_type bdd \
  --sample_ids "${SAMPLE_IDS[@]}" \
  --max_history_frames 41 \
  --height "$HEIGHT" \
  --width "$WIDTH" \
  --num_frames "$NUM_FRAMES" \
  --prompt_mode behavior \
  --skip_bad \
  --resume

echo "[2/4] Pixel metrics"
"$PYTHON" -m wan_ticft.evaluation.evaluate_video_reports \
  --manifest "$OUT/selected_manifest.jsonl" \
  --reports \
    ltx13b_fp8="$OUT/report.csv" \
    ltx2b_base="$EVAL_BASE/report.csv" \
    depth_a75="$EVAL_DEPTH/depth_layered_a75/report.csv" \
  --include_repeat_last \
  --offset_mode timestamp \
  --out_csv "${OUT}_eval.csv"

echo "[3/4] FID/FVD proxy"
CUDA_VISIBLE_DEVICES="$GPU_ID" TORCH_HOME="$ROOT/tools/torch_cache" \
"$PYTHON" -m wan_ticft.evaluation.audit_perceptual_video_distribution \
  --manifest "$OUT/selected_manifest.jsonl" \
  --report ltx13b_fp8="$OUT/report.csv" \
  --report ltx2b_base="$EVAL_BASE/report.csv" \
  --report depth_a75="$EVAL_DEPTH/depth_layered_a75/report.csv" \
  --out_csv "${OUT}_proxy.csv" \
  --clip_frames 8 \
  --frame_stride 1

echo "[4/4] Contact sheets"
mapfile -t GENERATED_SAMPLE_IDS < <("$PYTHON" - <<PY
import json
for line in open("$OUT/selected_manifest.jsonl", encoding="utf-8"):
    if line.strip():
        print(json.loads(line)["sample_id"])
PY
)
for sample_id in "${GENERATED_SAMPLE_IDS[@]}"; do
  "$PYTHON" -m wan_ticft.evaluation.make_report_comparison_sheet \
    --manifest "$OUT/selected_manifest.jsonl" \
    --reports \
      ltx13b_fp8="$OUT/report.csv" \
      ltx2b_base="$EVAL_BASE/report.csv" \
      depth_a75="$EVAL_DEPTH/depth_layered_a75/report.csv" \
    --sample_id "$sample_id" \
    --out_dir "${OUT}_compare_sheets" \
    --frames 8 \
    --tile_width 260
done

echo
echo "eval=${OUT}_eval.csv"
echo "proxy=${OUT}_proxy.csv"
echo "sheets=${OUT}_compare_sheets"
echo
echo "Gate:"
echo "  Continue to LIMIT=4 only if FP8 base has sane geometry and improves FID/FVD or pixel quality."
echo "  If full-res FP8 OOMs, retry HEIGHT=640 WIDTH=1088 before giving up."

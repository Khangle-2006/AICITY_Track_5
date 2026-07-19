#!/usr/bin/env bash
set -euo pipefail

cd /workspace

PYTHON=/opt/conda/envs/wan22/bin/python
ROOT=/workspace/experiments/wan_ticft
MANIFEST="$ROOT/ltxv_bdd25_gate/dense40_manifest.jsonl"
EVAL_DEPTH="$ROOT/bdd_supervised_ltx_depth_eval24/depth"
EVAL_BASE="$ROOT/ltxv_bdd25_gate"
LTX_REPO=/workspace/third_party/LTX-Video
LTX_PY=/workspace/experiments/wan_ticft/ltxv_env/bin/python
LTX_CFG=/workspace/wan_ticft/evaluation/ltxv_2b_gate.yaml

GPU_ID="${CUDA_VISIBLE_DEVICES:-2}"
GPU_ID="${GPU_ID%%,*}"
LIMIT="${LIMIT:-24}"
HEIGHT="${HEIGHT:-704}"
WIDTH="${WIDTH:-1216}"
NUM_FRAMES="${NUM_FRAMES:-121}"
FRAMEINIT_STRENGTH="${FRAMEINIT_STRENGTH:-0.35}"
FRAMEINIT_KERNEL="${FRAMEINIT_KERNEL:-9}"
FRAMEINIT_ANCHOR="${FRAMEINIT_ANCHOR:-last}"
OUT="$ROOT/ltx2b_frameinit_s${FRAMEINIT_STRENGTH//./}_k${FRAMEINIT_KERNEL}_bdd${LIMIT}"

echo "=== LTX 2B FrameInit BDD gate ==="
echo "out=$OUT"
echo "limit=$LIMIT resolution=${HEIGHT}x${WIDTH} frames=$NUM_FRAMES"
echo "FrameInit strength=$FRAMEINIT_STRENGTH kernel=$FRAMEINIT_KERNEL anchor=$FRAMEINIT_ANCHOR"

for path in "$MANIFEST" "$EVAL_BASE/report.csv" "$EVAL_DEPTH/depth_layered_a75/report.csv" "$LTX_CFG"; do
  [[ -s "$path" ]] || { echo "Missing input: $path" >&2; exit 2; }
done

echo "[1/4] Generate/resume LTX 2B FrameInit predictions"
CUDA_VISIBLE_DEVICES="$GPU_ID" \
LTX_FRAMEINIT=1 \
LTX_FRAMEINIT_STRENGTH="$FRAMEINIT_STRENGTH" \
LTX_FRAMEINIT_KERNEL="$FRAMEINIT_KERNEL" \
LTX_FRAMEINIT_ANCHOR="$FRAMEINIT_ANCHOR" \
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
    frameinit="$OUT/report.csv" \
    ltx2b_base="$EVAL_BASE/report.csv" \
    depth_a75="$EVAL_DEPTH/depth_layered_a75/report.csv" \
  --include_repeat_last \
  --offset_mode timestamp \
  --out_csv "${OUT}_eval.csv"

echo "[3/4] FID/FVD proxy"
CUDA_VISIBLE_DEVICES="$GPU_ID" TORCH_HOME="$ROOT/tools/torch_cache" \
"$PYTHON" -m wan_ticft.evaluation.audit_perceptual_video_distribution \
  --manifest "$OUT/selected_manifest.jsonl" \
  --report frameinit="$OUT/report.csv" \
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
      frameinit="$OUT/report.csv" \
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
echo "Continue only if frameinit improves FID/FVD or LPIPS-like visual quality without a large PSNR/SSIM collapse."


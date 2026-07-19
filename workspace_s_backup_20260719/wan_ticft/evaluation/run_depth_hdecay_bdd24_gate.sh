#!/usr/bin/env bash
set -euo pipefail

cd /workspace

PYTHON=/opt/conda/envs/wan22/bin/python
ROOT=/workspace/experiments/wan_ticft
GPU_ID="${CUDA_VISIBLE_DEVICES:-2}"
GPU_ID="${GPU_ID%%,*}"
BASE="$ROOT/ltxv_bdd25_gate"
DEPTH="$ROOT/bdd_supervised_ltx_depth_eval24/depth/depth_layered_a75/report.csv"
PHASE12="$ROOT/bdd24_phase12_convex_depth_rule/report.csv"
MODEL_CACHE="$ROOT/depth_layered_models"
START="${START:-20}"
END="${END:-80}"
FLOOR="${FLOOR:-0.35}"
OUT="$ROOT/depth_hdecay_a75_f${FLOOR}_s${START}_e${END}_bdd24"

echo "=== BDD24 depth temporal-decay alpha gate ==="
echo "alpha=0.75 floor=$FLOOR start=$START end=$END"
echo "out=$OUT"
/workspace/wan_ticft/evaluation/check_gpu_free.sh "$GPU_ID" 12000

for path in "$BASE/selected_manifest.jsonl" "$BASE/report.csv" "$DEPTH"; do
  [[ -s "$path" ]] || { echo "Missing input: $path" >&2; exit 2; }
done

echo "[1/4] Render depth hdecay variant"
CUDA_VISIBLE_DEVICES="$GPU_ID" HF_HOME="$MODEL_CACHE/hf_home" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
"$PYTHON" -m wan_ticft.evaluation.materialize_depth_layered_renderer \
  --manifest "$BASE/selected_manifest.jsonl" \
  --base_report "$BASE/report.csv" \
  --out_root "$OUT" \
  --model_cache "$MODEL_CACHE/huggingface" \
  --view_type bdd \
  --preserve_full_video \
  --limit 24 \
  --alphas 0.75 \
  --horizon_decay \
  --horizon_decay_start "$START" \
  --horizon_decay_end "$END" \
  --horizon_decay_alpha 0.75 \
  --horizon_decay_floor "$FLOOR" \
  --focal_scale 0.9 \
  --point_stride 2 \
  --splat_radius 1 \
  --min_matches 20 \
  --min_inliers 12 \
  --photo_sigma 32 \
  --segment_batch 4 \
  --pose_smoothing_window 7 \
  --filled_pose_scale 0.35 \
  --confidence_ema 0.65 \
  --clean

echo "[2/4] Pixel metrics"
REPORTS=(depth_a75="$DEPTH" hdecay="$OUT/depth_layered_hdecay/report.csv")
if [[ -s "$PHASE12" ]]; then
  REPORTS+=(phase12="$PHASE12")
fi
"$PYTHON" -m wan_ticft.evaluation.evaluate_video_reports \
  --manifest "$BASE/selected_manifest.jsonl" \
  --reports "${REPORTS[@]}" \
  --include_repeat_last \
  --offset_mode timestamp \
  --out_csv "${OUT}_eval.csv"

echo "[3/4] FID/FVD proxy"
PROXY_REPORTS=(--report depth_a75="$DEPTH" --report hdecay="$OUT/depth_layered_hdecay/report.csv")
if [[ -s "$PHASE12" ]]; then
  PROXY_REPORTS+=(--report phase12="$PHASE12")
fi
CUDA_VISIBLE_DEVICES="$GPU_ID" TORCH_HOME="$ROOT/tools/torch_cache" \
"$PYTHON" -m wan_ticft.evaluation.audit_perceptual_video_distribution \
  --manifest "$BASE/selected_manifest.jsonl" \
  "${PROXY_REPORTS[@]}" \
  --out_csv "${OUT}_proxy.csv" \
  --clip_frames 8 \
  --frame_stride 1

echo "[4/4] Contact sheets"
mapfile -t SAMPLE_IDS < <("$PYTHON" - <<PY
import json
for i, line in enumerate(open("$BASE/selected_manifest.jsonl", encoding="utf-8")):
    if i in {0, 5, 9, 13, 20, 23} and line.strip():
        print(json.loads(line)["sample_id"])
PY
)
for sample_id in "${SAMPLE_IDS[@]}"; do
  "$PYTHON" -m wan_ticft.evaluation.make_report_comparison_sheet \
    --manifest "$BASE/selected_manifest.jsonl" \
    --reports "${REPORTS[@]}" \
    --sample_id "$sample_id" \
    --out_dir "${OUT}_compare_sheets" \
    --frames 8 \
    --tile_width 220
done

echo
echo "eval=${OUT}_eval.csv"
echo "proxy=${OUT}_proxy.csv"
echo "sheets=${OUT}_compare_sheets"
echo
echo "Gate:"
echo "  Continue only if hdecay beats depth_a75 and remains competitive with phase12 on FID/FVD."

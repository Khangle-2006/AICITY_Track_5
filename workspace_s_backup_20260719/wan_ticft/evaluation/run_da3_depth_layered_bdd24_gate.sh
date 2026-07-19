#!/usr/bin/env bash
set -euo pipefail

cd /workspace

PYTHON=/opt/conda/envs/wan22/bin/python
ROOT=/workspace/experiments/wan_ticft
GPU_ID="${CUDA_VISIBLE_DEVICES:-2}"
GPU_ID="${GPU_ID%%,*}"
BASE="$ROOT/ltxv_bdd25_gate"
OLD_DEPTH="$ROOT/bdd_supervised_ltx_depth_eval24/depth/depth_layered_a75/report.csv"
MODEL_CACHE="$ROOT/depth_anything3_models"
BACKEND="${BACKEND:-da3_mono}"
DA3_MODEL_ID="${DA3_MODEL_ID:-}"
DA3_PROCESS_RES="${DA3_PROCESS_RES:-504}"
OUT="${OUT:-$ROOT/${BACKEND}_depth_layered_a75_bdd24_r${DA3_PROCESS_RES}}"

echo "=== BDD24 Depth Anything 3 depth-layered gate ==="
echo "backend=$BACKEND"
echo "model_id=${DA3_MODEL_ID:-auto}"
echo "process_res=$DA3_PROCESS_RES"
echo "out=$OUT"

for path in "$BASE/selected_manifest.jsonl" "$BASE/report.csv" "$OLD_DEPTH"; do
  [[ -s "$path" ]] || { echo "Missing input: $path" >&2; exit 2; }
done

/workspace/wan_ticft/evaluation/check_gpu_free.sh "$GPU_ID" 12000

if ! "$PYTHON" - <<'PY'
import importlib.util
raise SystemExit(0 if importlib.util.find_spec("depth_anything_3") else 1)
PY
then
  echo "Depth Anything 3 is not installed in wan22." >&2
  echo "Run: bash /workspace/wan_ticft/evaluation/setup_depth_anything3.sh" >&2
  exit 2
fi

DA3_ARGS=()
if [[ -n "$DA3_MODEL_ID" ]]; then
  DA3_ARGS+=(--da3_model_id "$DA3_MODEL_ID")
fi

echo "[1/4] Render DA3 depth layered a75"
CUDA_VISIBLE_DEVICES="$GPU_ID" \
HF_HOME="$MODEL_CACHE/hf_home" \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
"$PYTHON" -m wan_ticft.evaluation.materialize_depth_layered_renderer \
  --manifest "$BASE/selected_manifest.jsonl" \
  --base_report "$BASE/report.csv" \
  --out_root "$OUT" \
  --model_cache "$MODEL_CACHE/huggingface" \
  --view_type bdd \
  --preserve_full_video \
  --limit 24 \
  --depth_backend "$BACKEND" \
  "${DA3_ARGS[@]}" \
  --da3_process_res "$DA3_PROCESS_RES" \
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
  --confidence_ema 0.65 \
  --clean

echo "[2/4] Pixel metrics"
"$PYTHON" -m wan_ticft.evaluation.evaluate_video_reports \
  --manifest "$BASE/selected_manifest.jsonl" \
  --reports \
    depth_a75="$OLD_DEPTH" \
    da3_a75="$OUT/depth_layered_a75/report.csv" \
  --include_repeat_last \
  --offset_mode timestamp \
  --out_csv "${OUT}_eval.csv"

echo "[3/4] FID/FVD proxy"
CUDA_VISIBLE_DEVICES="$GPU_ID" TORCH_HOME="$ROOT/tools/torch_cache" \
"$PYTHON" -m wan_ticft.evaluation.audit_perceptual_video_distribution \
  --manifest "$BASE/selected_manifest.jsonl" \
  --report depth_a75="$OLD_DEPTH" \
  --report da3_a75="$OUT/depth_layered_a75/report.csv" \
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
    --reports \
      depth_a75="$OLD_DEPTH" \
      da3_a75="$OUT/depth_layered_a75/report.csv" \
    --sample_id "$sample_id" \
    --out_dir "${OUT}_compare_sheets" \
    --frames 8 \
    --tile_width 220
done

echo
echo "eval=${OUT}_eval.csv"
echo "proxy=${OUT}_proxy.csv"
echo "diagnostics=$OUT/diagnostics.csv"
echo "sheets=${OUT}_compare_sheets"
echo
echo "Gate:"
echo "  Continue only if da3_a75 improves FID/FVD or pixel without introducing doubled-road/vehicle edges."
echo "  If da3_mono is close, rerun with BACKEND=da3_metric before deciding."

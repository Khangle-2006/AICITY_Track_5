#!/usr/bin/env bash
set -euo pipefail

cd /workspace

PYTHON=/opt/conda/envs/wan22/bin/python
ROOT=/workspace/experiments/wan_ticft
LIMIT="${LIMIT:-4}"
MANIFEST="$ROOT/ltxv_bdd25_gate/dense40_manifest.jsonl"
BASE_REPORT="$ROOT/cnn_selector_perceptual_tradeoff_gate/cnn_ltxmix65/report.csv"
OUT="${OUT:-$ROOT/depth_layered_renderer_temporal_bdd${LIMIT}}"
SELECTED_MANIFEST="$OUT/selected_manifest.jsonl"
MODEL_CACHE="$ROOT/depth_layered_models"
GPU_ID="${CUDA_VISIBLE_DEVICES:-3}"
GPU_ID="${GPU_ID%%,*}"

echo "=== Temporally smoothed depth-aware 3D splatting BDD${LIMIT} gate ==="
echo "Depth: Intel DPT Hybrid MiDaS (downloaded once into $MODEL_CACHE)"
echo "Actors: cached torchvision DeepLabV3-ResNet50"
echo "Base owns actors, holes, and failed poses; splatting only supplies trusted static texture."

for path in "$MANIFEST" "$BASE_REPORT"; do
  if [[ ! -f "$path" ]]; then
    echo "Missing input: $path" >&2
    exit 2
  fi
done
/workspace/wan_ticft/evaluation/check_gpu_free.sh "$GPU_ID" 12000
FREE_GIB=$(df --output=avail -BG /workspace | tail -1 | tr -dc '0-9')
if (( FREE_GIB < 8 )); then
  echo "Need at least 8 GiB free on /workspace; found ${FREE_GIB} GiB." >&2
  exit 2
fi

CUDA_VISIBLE_DEVICES="$GPU_ID" \
HF_HOME="$MODEL_CACHE/hf_home" \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
"$PYTHON" -m wan_ticft.evaluation.materialize_depth_layered_renderer \
  --manifest "$MANIFEST" \
  --base_report "$BASE_REPORT" \
  --out_root "$OUT" \
  --model_cache "$MODEL_CACHE/huggingface" \
  --limit "$LIMIT" \
  --alphas 0.25 0.50 0.75 \
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

"$PYTHON" -m wan_ticft.evaluation.evaluate_video_reports \
  --manifest "$SELECTED_MANIFEST" \
  --reports \
    cnn_ltxmix65="$BASE_REPORT" \
    depth_a25="$OUT/depth_layered_a25/report.csv" \
    depth_a50="$OUT/depth_layered_a50/report.csv" \
    depth_a75="$OUT/depth_layered_a75/report.csv" \
  --include_repeat_last \
  --offset_mode timestamp \
  --out_csv "${OUT}_eval.csv"

CUDA_VISIBLE_DEVICES="$GPU_ID" \
TORCH_HOME="$ROOT/tools/torch_cache" \
"$PYTHON" -m wan_ticft.evaluation.audit_perceptual_video_distribution \
  --manifest "$SELECTED_MANIFEST" \
  --report cnn_ltxmix65="$BASE_REPORT" \
  --report depth_a25="$OUT/depth_layered_a25/report.csv" \
  --report depth_a50="$OUT/depth_layered_a50/report.csv" \
  --report depth_a75="$OUT/depth_layered_a75/report.csv" \
  --out_csv "${OUT}_proxy.csv" \
  --clip_frames 16 \
  --frame_stride 4

mapfile -t SAMPLE_IDS < <("$PYTHON" - <<PY
import json
count = 0
for line in open("$SELECTED_MANIFEST", encoding="utf-8"):
    if line.strip():
        row = json.loads(line)
        if row.get("view_type") == "bdd" and count < int("$LIMIT"):
            print(row["sample_id"])
            count += 1
PY
)
SHEET_ARGS=()
for sample_id in "${SAMPLE_IDS[@]}"; do
  SHEET_ARGS+=(--sample_id "$sample_id")
done

"$PYTHON" -m wan_ticft.evaluation.make_report_comparison_sheet \
  --manifest "$SELECTED_MANIFEST" \
  --reports \
    cnn_ltxmix65="$BASE_REPORT" \
    depth_a25="$OUT/depth_layered_a25/report.csv" \
    depth_a50="$OUT/depth_layered_a50/report.csv" \
    depth_a75="$OUT/depth_layered_a75/report.csv" \
  "${SHEET_ARGS[@]}" \
  --out_dir "${OUT}_compare_sheets" \
  --frames 8 \
  --tile_width 220

echo
echo "eval=${OUT}_eval.csv"
echo "proxy=${OUT}_proxy.csv"
echo "diagnostics=$OUT/diagnostics.csv"
echo "sheets=${OUT}_compare_sheets"
echo
echo "Gate:"
echo "  - Median pose inliers >= 12 and mean coverage >= 0.20; otherwise renderer is inactive."
echo "  - Continue only if one alpha improves aggregate PSNR >= 0.10, does not lower SSIM,"
echo "    does not worsen FVD, keeps FID increase <= 0.15, and sheets show no doubled edges."

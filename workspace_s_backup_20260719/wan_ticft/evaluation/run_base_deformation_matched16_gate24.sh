#!/usr/bin/env bash
set -euo pipefail

cd /workspace

PYTHON=/opt/conda/envs/wan22/bin/python
ROOT=/workspace/experiments/wan_ticft
TRAIN_ROOT="$ROOT/matched_cnn_ltxmix65_train16"
TRAIN_MANIFEST="$TRAIN_ROOT/dense40_manifest.jsonl"
TRAIN_REPORT="$TRAIN_ROOT/cnn_selector/cnn_ltxmix65/report.csv"
EVAL_MANIFEST="$ROOT/ltxv_bdd25_gate/dense40_manifest.jsonl"
BASE_REPORT="$ROOT/cnn_selector_perceptual_tradeoff_gate/cnn_ltxmix65/report.csv"
OUT="$ROOT/base_deformation_matched16_gate24"
GPU_ID="${CUDA_VISIBLE_DEVICES:-2}"
GPU_ID="${GPU_ID%%,*}"

echo "=== Base-conditioned temporal deformation: matched16 -> BDD24 ==="
echo "This model can only warp original cnn_ltxmix65 pixels by at most 3 low-resolution pixels."
/workspace/wan_ticft/evaluation/check_gpu_free.sh "$GPU_ID" 20000

for path in "$TRAIN_MANIFEST" "$TRAIN_REPORT" "$EVAL_MANIFEST" "$BASE_REPORT"; do
  [[ -s "$path" ]] || { echo "Missing input: $path" >&2; exit 2; }
done

CUDA_VISIBLE_DEVICES="$GPU_ID" \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
TORCH_HOME="$ROOT/tools/torch_cache" \
"$PYTHON" -m wan_ticft.evaluation.train_base_deformation_refiner \
  --train_manifest "$TRAIN_MANIFEST" \
  --train_report "$TRAIN_REPORT" \
  --eval_manifest "$EVAL_MANIFEST" \
  --eval_report "$BASE_REPORT" \
  --out_dir "$OUT" \
  --train_limit 16 \
  --eval_limit 24 \
  --width 320 \
  --height 180 \
  --window 8 \
  --steps 800 \
  --model_width 32 \
  --max_flow_px 3.0 \
  --alphas 0.25 0.50 0.75 1.00 \
  --clean

REPORTS=(cnn_ltxmix65="$BASE_REPORT")
PROXY_REPORTS=(--report cnn_ltxmix65="$BASE_REPORT")
SHEET_REPORTS=(cnn_ltxmix65="$BASE_REPORT")
for tag in a025 a050 a075 a100; do
  REPORTS+=(deform_"$tag"="$OUT/report_${tag}.csv")
  PROXY_REPORTS+=(--report deform_"$tag"="$OUT/report_${tag}.csv")
  SHEET_REPORTS+=(deform_"$tag"="$OUT/report_${tag}.csv")
done

"$PYTHON" -m wan_ticft.evaluation.evaluate_video_reports \
  --manifest "$EVAL_MANIFEST" \
  --reports "${REPORTS[@]}" \
  --include_repeat_last \
  --offset_mode timestamp \
  --out_csv "${OUT}_eval.csv"

CUDA_VISIBLE_DEVICES="$GPU_ID" TORCH_HOME="$ROOT/tools/torch_cache" \
"$PYTHON" -m wan_ticft.evaluation.audit_perceptual_video_distribution \
  --manifest "$EVAL_MANIFEST" \
  "${PROXY_REPORTS[@]}" \
  --out_csv "${OUT}_proxy.csv" \
  --clip_frames 16 \
  --frame_stride 4

mapfile -t SAMPLE_IDS < <("$PYTHON" - <<PY
import json
for line in open("$EVAL_MANIFEST", encoding="utf-8"):
    if line.strip():
        row = json.loads(line)
        if row.get("view_type") == "bdd":
            print(row["sample_id"])
PY
)
SHEET_ARGS=()
for sample_id in "${SAMPLE_IDS[@]:0:24}"; do
  SHEET_ARGS+=(--sample_id "$sample_id")
done

"$PYTHON" -m wan_ticft.evaluation.make_report_comparison_sheet \
  --manifest "$EVAL_MANIFEST" \
  --reports "${SHEET_REPORTS[@]}" \
  "${SHEET_ARGS[@]}" \
  --out_dir "${OUT}_compare_sheets" \
  --frames 8 \
  --tile_width 200

echo
echo "eval=${OUT}_eval.csv"
echo "proxy=${OUT}_proxy.csv"
echo "diagnostics=$OUT/diagnostics.csv"
echo "sheets=${OUT}_compare_sheets"
echo
echo "Gate: continue only if one alpha improves FVD by at least 0.30, does not worsen FID,"
echo "and loses no more than 0.05 PSNR / 0.001 SSIM. Otherwise close deformation learning."

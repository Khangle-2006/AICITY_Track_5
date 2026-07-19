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
OUT="$ROOT/highres_temporal_detail_matched16_gate24"
GPU_ID="${CUDA_VISIBLE_DEVICES:-3}"
GPU_ID="${GPU_ID%%,*}"

echo "=== High-resolution temporal detail refiner: matched16 -> BDD24 ==="
/workspace/wan_ticft/evaluation/check_gpu_free.sh "$GPU_ID" 30000

CUDA_VISIBLE_DEVICES="$GPU_ID" \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
"$PYTHON" -m wan_ticft.evaluation.train_highres_temporal_detail_refiner \
  --train_manifest "$TRAIN_MANIFEST" \
  --train_report "$TRAIN_REPORT" \
  --eval_manifest "$EVAL_MANIFEST" \
  --eval_report "$BASE_REPORT" \
  --out_dir "$OUT" \
  --train_limit 16 \
  --eval_limit 24 \
  --train_width 640 \
  --train_height 360 \
  --patch 160 \
  --batch_size 6 \
  --epochs 6 \
  --steps_per_epoch 80 \
  --model_width 48 \
  --residual_scale 0.04 \
  --clean

"$PYTHON" -m wan_ticft.evaluation.evaluate_video_reports \
  --manifest "$EVAL_MANIFEST" \
  --reports cnn_ltxmix65="$BASE_REPORT" highres_detail="$OUT/report.csv" \
  --include_repeat_last \
  --offset_mode timestamp \
  --out_csv "${OUT}_eval.csv"

CUDA_VISIBLE_DEVICES="$GPU_ID" \
TORCH_HOME="$ROOT/tools/torch_cache" \
"$PYTHON" -m wan_ticft.evaluation.audit_perceptual_video_distribution \
  --manifest "$EVAL_MANIFEST" \
  --report cnn_ltxmix65="$BASE_REPORT" \
  --report highres_detail="$OUT/report.csv" \
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
  --reports cnn_ltxmix65="$BASE_REPORT" highres_detail="$OUT/report.csv" \
  "${SHEET_ARGS[@]}" \
  --out_dir "${OUT}_compare_sheets" \
  --frames 8 \
  --tile_width 220

echo "eval=${OUT}_eval.csv"
echo "proxy=${OUT}_proxy.csv"
echo "sheets=${OUT}_compare_sheets"

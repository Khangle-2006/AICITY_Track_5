#!/usr/bin/env bash
set -euo pipefail
cd /workspace

PYTHON=/opt/conda/envs/wan22/bin/python
ROOT=/workspace/experiments/wan_ticft
EVAL_MANIFEST="$ROOT/ltxv_bdd25_gate/dense40_manifest.jsonl"
BASE_REPORT="$ROOT/cnn_selector_perceptual_tradeoff_gate/cnn_ltxmix65/report.csv"
OUT="$ROOT/highres_temporal_detail_matched16_gate24"

"$PYTHON" -m wan_ticft.evaluation.evaluate_video_reports \
  --manifest "$EVAL_MANIFEST" \
  --reports cnn_ltxmix65="$BASE_REPORT" highres_detail="$OUT/report.csv" \
  --include_repeat_last \
  --offset_mode timestamp \
  --out_csv "${OUT}_eval.csv"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-3}" \
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

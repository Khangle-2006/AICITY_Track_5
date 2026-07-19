#!/usr/bin/env bash
set -euo pipefail

cd /workspace

PYTHON=/opt/conda/envs/wan22/bin/python
ROOT=/workspace/experiments/wan_ticft
GPU_ID="${CUDA_VISIBLE_DEVICES:-0}"
GPU_ID="${GPU_ID%%,*}"
TRAIN_LIMIT="${TRAIN_LIMIT:-48}"

TRAIN_BASE="$ROOT/bdd_supervised_ltx_depth_train${TRAIN_LIMIT}/ltx_base"
TRAIN_MANIFEST="$TRAIN_BASE/selected_manifest_repaired_conditions.jsonl"
TRAIN_REPEAT="$ROOT/bdd_supervised_repeat_depth_train${TRAIN_LIMIT}/repeat_last"
TRAIN_DEPTH="$ROOT/bdd_supervised_ltx_depth_train${TRAIN_LIMIT}/depth"

EVAL_BASE="$ROOT/ltxv_bdd25_gate"
EVAL_REPEAT="$ROOT/bdd_supervised_repeat_depth_eval24/repeat_last"
EVAL_DEPTH="$ROOT/bdd_supervised_ltx_depth_eval24/depth"
OUT="$ROOT/bdd_supervised_convex_repeat_depth_train${TRAIN_LIMIT}_eval24"

echo "=== BDD supervised bounded convex repeat/depth blend: train${TRAIN_LIMIT} -> BDD24 ==="
echo "Unlike v2, this cannot extrapolate outside the repeat_last/depth_a75 segment."
/workspace/wan_ticft/evaluation/check_gpu_free.sh "$GPU_ID" 8000

for path in \
  "$TRAIN_BASE/selected_manifest.jsonl" \
  "$TRAIN_BASE/history_videos" \
  "$TRAIN_DEPTH/depth_layered_a75/report.csv" \
  "$EVAL_BASE/selected_manifest.jsonl" \
  "$EVAL_DEPTH/depth_layered_a75/report.csv"; do
  [[ -e "$path" ]] || { echo "Missing input: $path" >&2; exit 2; }
done

echo "[1/6] Repair train condition/target frames"
"$PYTHON" -m wan_ticft.evaluation.repair_ltx_selected_manifest_conditions \
  --manifest "$TRAIN_BASE/selected_manifest.jsonl" \
  --history_dir "$TRAIN_BASE/history_videos" \
  --frames_root "$TRAIN_BASE/repaired_condition_frames" \
  --out_manifest "$TRAIN_MANIFEST" \
  --skip_bad \
  --clean

echo "[2/6] Materialize train repeat-last videos"
"$PYTHON" -m wan_ticft.evaluation.materialize_repeat_last_videos \
  --manifest "$TRAIN_MANIFEST" \
  --reference_report "$TRAIN_DEPTH/depth_layered_a75/report.csv" \
  --out_dir "$TRAIN_REPEAT" \
  --view_type bdd

echo "[3/6] Materialize eval repeat-last videos"
"$PYTHON" -m wan_ticft.evaluation.materialize_repeat_last_videos \
  --manifest "$EVAL_BASE/selected_manifest.jsonl" \
  --reference_report "$EVAL_DEPTH/depth_layered_a75/report.csv" \
  --out_dir "$EVAL_REPEAT" \
  --view_type bdd \
  --limit 24

echo "[4/6] Train bounded convex blend and infer BDD24"
CUDA_VISIBLE_DEVICES="$GPU_ID" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
"$PYTHON" -m wan_ticft.evaluation.train_supervised_convex_blend \
  --train_manifest "$TRAIN_MANIFEST" \
  --train_repeat_report "$TRAIN_REPEAT/report.csv" \
  --train_depth_report "$TRAIN_DEPTH/depth_layered_a75/report.csv" \
  --eval_manifest "$EVAL_BASE/selected_manifest.jsonl" \
  --eval_repeat_report "$EVAL_REPEAT/report.csv" \
  --eval_depth_report "$EVAL_DEPTH/depth_layered_a75/report.csv" \
  --out_dir "$OUT" \
  --view_type bdd \
  --train_limit "$TRAIN_LIMIT" \
  --train_width 640 \
  --train_height 360 \
  --patch 128 \
  --batch_size 8 \
  --epochs 10 \
  --steps_per_epoch 180 \
  --model_width 32 \
  --model_blocks 5 \
  --init_alpha 0.90 \
  --lr 3e-4 \
  --lambda_edge 0.08 \
  --lambda_depth_prior 0.01 \
  --clean

echo "[5/6] Pixel and FID/FVD proxy gate"
"$PYTHON" -m wan_ticft.evaluation.evaluate_video_reports \
  --manifest "$EVAL_BASE/selected_manifest.jsonl" \
  --reports \
    repeat_video="$EVAL_REPEAT/report.csv" \
    depth_a75="$EVAL_DEPTH/depth_layered_a75/report.csv" \
    convex_blend="$OUT/report.csv" \
  --include_repeat_last \
  --offset_mode timestamp \
  --out_csv "${OUT}_eval.csv"

CUDA_VISIBLE_DEVICES="$GPU_ID" TORCH_HOME="$ROOT/tools/torch_cache" \
"$PYTHON" -m wan_ticft.evaluation.audit_perceptual_video_distribution \
  --manifest "$EVAL_BASE/selected_manifest.jsonl" \
  --report repeat_video="$EVAL_REPEAT/report.csv" \
  --report depth_a75="$EVAL_DEPTH/depth_layered_a75/report.csv" \
  --report convex_blend="$OUT/report.csv" \
  --out_csv "${OUT}_proxy.csv" \
  --clip_frames 8 \
  --frame_stride 1

echo "[6/6] Contact sheets"
mapfile -t SAMPLE_IDS < <("$PYTHON" - <<PY
import json
for i, line in enumerate(open("$EVAL_BASE/selected_manifest.jsonl", encoding="utf-8")):
    if i in {0, 6, 9, 14, 19, 23} and line.strip():
        print(json.loads(line)["sample_id"])
PY
)
for sample_id in "${SAMPLE_IDS[@]}"; do
  "$PYTHON" -m wan_ticft.evaluation.make_report_comparison_sheet \
    --manifest "$EVAL_BASE/selected_manifest.jsonl" \
    --reports \
      repeat_video="$EVAL_REPEAT/report.csv" \
      depth_a75="$EVAL_DEPTH/depth_layered_a75/report.csv" \
      convex_blend="$OUT/report.csv" \
    --sample_id "$sample_id" \
    --out_dir "${OUT}_compare_sheets" \
    --frames 8 \
    --tile_width 220
done

echo
echo "eval=${OUT}_eval.csv"
echo "proxy=${OUT}_proxy.csv"
echo "history=$OUT/history.csv"
echo "alpha=$OUT/alpha_stats.csv"
echo "sheets=${OUT}_compare_sheets"
echo
echo "Gate:"
echo "  Promote only if convex_blend gains PSNR over depth_a75 while preserving FVD/FID."
echo "  If alpha collapses near 0 or 1 on every sample, this branch is not learning useful routing."

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
OUT="$ROOT/bdd_supervised_repeat_depth_blend_train${TRAIN_LIMIT}_eval24"

echo "=== BDD supervised repeat-last/depth blend: train${TRAIN_LIMIT} -> BDD24 ==="
echo "Learns when to keep depth_a75 and when to pull static scenes toward repeat-last."
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
  --view_type bdd \
  --clean

echo "[3/6] Materialize eval repeat-last videos"
"$PYTHON" -m wan_ticft.evaluation.materialize_repeat_last_videos \
  --manifest "$EVAL_BASE/selected_manifest.jsonl" \
  --reference_report "$EVAL_DEPTH/depth_layered_a75/report.csv" \
  --out_dir "$EVAL_REPEAT" \
  --view_type bdd \
  --limit 24 \
  --clean

echo "[4/6] Train supervised repeat-depth blend and infer BDD24"
CUDA_VISIBLE_DEVICES="$GPU_ID" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
"$PYTHON" -m wan_ticft.evaluation.train_supervised_residual_gate \
  --train_manifest "$TRAIN_MANIFEST" \
  --train_base_report "$TRAIN_REPEAT/report.csv" \
  --train_candidate_report "$TRAIN_DEPTH/depth_layered_a75/report.csv" \
  --eval_manifest "$EVAL_BASE/selected_manifest.jsonl" \
  --eval_base_report "$EVAL_REPEAT/report.csv" \
  --eval_candidate_report "$EVAL_DEPTH/depth_layered_a75/report.csv" \
  --out_dir "$OUT" \
  --view_type bdd \
  --train_limit "$TRAIN_LIMIT" \
  --train_width 640 \
  --train_height 360 \
  --patch 128 \
  --batch_size 8 \
  --epochs 12 \
  --steps_per_epoch 200 \
  --model_width 48 \
  --model_blocks 6 \
  --max_delta 1.50 \
  --lr 8e-4 \
  --lambda_edge 0.08 \
  --lambda_identity 0.0 \
  --clean

echo "[5/6] Pixel and FID/FVD proxy gate"
"$PYTHON" -m wan_ticft.evaluation.evaluate_video_reports \
  --manifest "$EVAL_BASE/selected_manifest.jsonl" \
  --reports \
    repeat_video="$EVAL_REPEAT/report.csv" \
    depth_a75="$EVAL_DEPTH/depth_layered_a75/report.csv" \
    supervised_blend="$OUT/report.csv" \
  --include_repeat_last \
  --offset_mode timestamp \
  --out_csv "${OUT}_eval.csv"

CUDA_VISIBLE_DEVICES="$GPU_ID" TORCH_HOME="$ROOT/tools/torch_cache" \
"$PYTHON" -m wan_ticft.evaluation.audit_perceptual_video_distribution \
  --manifest "$EVAL_BASE/selected_manifest.jsonl" \
  --report repeat_video="$EVAL_REPEAT/report.csv" \
  --report depth_a75="$EVAL_DEPTH/depth_layered_a75/report.csv" \
  --report supervised_blend="$OUT/report.csv" \
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
      supervised_blend="$OUT/report.csv" \
    --sample_id "$sample_id" \
    --out_dir "${OUT}_compare_sheets" \
    --frames 8 \
    --tile_width 220
done

echo
echo "eval=${OUT}_eval.csv"
echo "proxy=${OUT}_proxy.csv"
echo "history=$OUT/history.csv"
echo "sheets=${OUT}_compare_sheets"
echo
echo "Gate:"
echo "  Continue only if supervised_blend improves PSNR materially over depth_a75,"
echo "  keeps SSIM within 0.002, and does not worsen FID/FVD proxy."

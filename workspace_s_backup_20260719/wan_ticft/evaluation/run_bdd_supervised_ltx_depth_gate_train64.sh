#!/usr/bin/env bash
set -euo pipefail

cd /workspace

PYTHON=/opt/conda/envs/wan22/bin/python
ROOT=/workspace/experiments/wan_ticft
GPU_ID="${CUDA_VISIBLE_DEVICES:-0}"
GPU_ID="${GPU_ID%%,*}"
TRAIN_LIMIT="${TRAIN_LIMIT:-48}"
TRAIN_SOURCE="${TRAIN_SOURCE:-$ROOT/manifests/train_ticft_balanced_300_frames.jsonl}"
TRAIN_BASE="$ROOT/bdd_supervised_ltx_depth_train${TRAIN_LIMIT}/ltx_base"
TRAIN_REPAIRED_MANIFEST="$TRAIN_BASE/selected_manifest_repaired_conditions.jsonl"
TRAIN_REPAIRED_FRAMES="$TRAIN_BASE/repaired_condition_frames"
TRAIN_DEPTH="$ROOT/bdd_supervised_ltx_depth_train${TRAIN_LIMIT}/depth"
EVAL_BASE="$ROOT/ltxv_bdd25_gate"
EVAL_DEPTH="$ROOT/bdd_supervised_ltx_depth_eval24/depth"
OUT="$ROOT/bdd_supervised_ltx_depth_gate_train${TRAIN_LIMIT}_eval24"
MODEL_CACHE="$ROOT/depth_layered_models"
LTX_REPO=/workspace/third_party/LTX-Video
LTX_PY=/workspace/experiments/wan_ticft/ltxv_env/bin/python
LTX_CFG=/workspace/wan_ticft/evaluation/ltxv_2b_gate.yaml

echo "=== BDD supervised LTX-depth residual gate: train${TRAIN_LIMIT} -> BDD24 ==="
echo "This is a real supervised gate, not a submission tweak."
echo "It trains only after materializing matched train base/candidate predictions."
/workspace/wan_ticft/evaluation/check_gpu_free.sh "$GPU_ID" 18000

for path in "$TRAIN_SOURCE" "$EVAL_BASE/selected_manifest.jsonl" "$EVAL_BASE/report.csv"; do
  [[ -s "$path" ]] || { echo "Missing input: $path" >&2; exit 2; }
done

echo "[1/6] Generate/resume LTX train${TRAIN_LIMIT} BDD base"
if [[ -s "$TRAIN_BASE/selected_manifest.jsonl" && -s "$TRAIN_BASE/report.csv" ]]; then
  echo "  found existing LTX train base; skip generation"
else
  CUDA_VISIBLE_DEVICES="$GPU_ID" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  "$PYTHON" -m wan_ticft.evaluation.run_ltxv_val_gate \
    --manifest "$TRAIN_SOURCE" \
    --out_dir "$TRAIN_BASE" \
    --ltx_repo "$LTX_REPO" \
    --ltx_python "$LTX_PY" \
    --pipeline_config "$LTX_CFG" \
    --limit "$TRAIN_LIMIT" \
    --view_type bdd \
    --max_history_frames 41 \
    --prompt_mode behavior \
    --skip_bad \
    --resume
fi

echo "[1b/6] Repair train manifest condition frames from LTX history videos"
"$PYTHON" -m wan_ticft.evaluation.repair_ltx_selected_manifest_conditions \
  --manifest "$TRAIN_BASE/selected_manifest.jsonl" \
  --history_dir "$TRAIN_BASE/history_videos" \
  --frames_root "$TRAIN_REPAIRED_FRAMES" \
  --out_manifest "$TRAIN_REPAIRED_MANIFEST" \
  --skip_bad \
  --clean

echo "[2/6] Generate train depth proposals over LTX base"
if [[ -s "$TRAIN_DEPTH/depth_layered_a00/report.csv" && -s "$TRAIN_DEPTH/depth_layered_a75/report.csv" ]]; then
  echo "  found existing train depth reports; skip generation"
else
  CUDA_VISIBLE_DEVICES="$GPU_ID" HF_HOME="$MODEL_CACHE/hf_home" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  "$PYTHON" -m wan_ticft.evaluation.materialize_depth_layered_renderer \
    --manifest "$TRAIN_REPAIRED_MANIFEST" \
    --base_report "$TRAIN_BASE/report.csv" \
    --out_root "$TRAIN_DEPTH" \
    --model_cache "$MODEL_CACHE/huggingface" \
    --view_type bdd \
    --preserve_full_video \
    --limit "$TRAIN_LIMIT" \
    --alphas 0.00 0.75 \
    --focal_scale 0.9 \
    --point_stride 2 \
    --splat_radius 1 \
    --min_matches 20 \
    --min_inliers 12 \
    --photo_sigma 32 \
    --segment_batch 4 \
    --pose_smoothing_window 7 \
    --filled_pose_scale 0.35 \
    --confidence_ema 0.65
fi

echo "[3/6] Generate eval depth proposals over existing BDD24 LTX base"
if [[ -s "$EVAL_DEPTH/depth_layered_a00/report.csv" && -s "$EVAL_DEPTH/depth_layered_a75/report.csv" ]]; then
  echo "  found existing eval depth reports; skip generation"
else
  CUDA_VISIBLE_DEVICES="$GPU_ID" HF_HOME="$MODEL_CACHE/hf_home" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  "$PYTHON" -m wan_ticft.evaluation.materialize_depth_layered_renderer \
    --manifest "$EVAL_BASE/selected_manifest.jsonl" \
    --base_report "$EVAL_BASE/report.csv" \
    --out_root "$EVAL_DEPTH" \
    --model_cache "$MODEL_CACHE/huggingface" \
    --view_type bdd \
    --preserve_full_video \
    --limit 24 \
    --alphas 0.00 0.75 \
    --focal_scale 0.9 \
    --point_stride 2 \
    --splat_radius 1 \
    --min_matches 20 \
    --min_inliers 12 \
    --photo_sigma 32 \
    --segment_batch 4 \
    --pose_smoothing_window 7 \
    --filled_pose_scale 0.35 \
    --confidence_ema 0.65
fi

echo "[4/6] Train supervised residual gate and infer BDD24"
CUDA_VISIBLE_DEVICES="$GPU_ID" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
"$PYTHON" -m wan_ticft.evaluation.train_supervised_residual_gate \
  --train_manifest "$TRAIN_REPAIRED_MANIFEST" \
  --train_base_report "$TRAIN_DEPTH/depth_layered_a00/report.csv" \
  --train_candidate_report "$TRAIN_DEPTH/depth_layered_a75/report.csv" \
  --eval_manifest "$EVAL_BASE/selected_manifest.jsonl" \
  --eval_base_report "$EVAL_DEPTH/depth_layered_a00/report.csv" \
  --eval_candidate_report "$EVAL_DEPTH/depth_layered_a75/report.csv" \
  --out_dir "$OUT" \
  --view_type bdd \
  --train_limit "$TRAIN_LIMIT" \
  --train_width 640 \
  --train_height 360 \
  --patch 128 \
  --batch_size 8 \
  --epochs 8 \
  --steps_per_epoch 150 \
  --model_width 32 \
  --model_blocks 5 \
  --max_delta 0.75 \
  --lambda_edge 0.10 \
  --lambda_identity 0.02 \
  --clean

echo "[5/6] Pixel and FID/FVD proxy gate"
"$PYTHON" -m wan_ticft.evaluation.evaluate_video_reports \
  --manifest "$EVAL_BASE/selected_manifest.jsonl" \
  --reports \
    ltx_base="$EVAL_DEPTH/depth_layered_a00/report.csv" \
    depth_a75="$EVAL_DEPTH/depth_layered_a75/report.csv" \
    supervised_gate="$OUT/report.csv" \
  --include_repeat_last \
  --offset_mode timestamp \
  --out_csv "${OUT}_eval.csv"

CUDA_VISIBLE_DEVICES="$GPU_ID" TORCH_HOME="$ROOT/tools/torch_cache" \
"$PYTHON" -m wan_ticft.evaluation.audit_perceptual_video_distribution \
  --manifest "$EVAL_BASE/selected_manifest.jsonl" \
  --report ltx_base="$EVAL_DEPTH/depth_layered_a00/report.csv" \
  --report depth_a75="$EVAL_DEPTH/depth_layered_a75/report.csv" \
  --report supervised_gate="$OUT/report.csv" \
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
      ltx_base="$EVAL_DEPTH/depth_layered_a00/report.csv" \
      depth_a75="$EVAL_DEPTH/depth_layered_a75/report.csv" \
      supervised_gate="$OUT/report.csv" \
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
echo "Promotion gate:"
echo "  Continue only if supervised_gate beats depth_a75 by >= +0.5 PSNR or clearly lowers FVD,"
echo "  without SSIM/FID collapse. Otherwise close supervised gate v1."

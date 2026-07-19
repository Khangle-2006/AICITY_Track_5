#!/usr/bin/env bash
set -euo pipefail

cd /workspace

PYTHON=/opt/conda/envs/wan22/bin/python
ROOT=/workspace/experiments/wan_ticft
SOURCE_MANIFEST="$ROOT/manifests/train_ticft_balanced_300_frames.jsonl"
TRAIN_ROOT="$ROOT/supervised_depth_gate_train32_data"
TRAIN_MANIFEST="$TRAIN_ROOT/train_vehicle32_manifest.jsonl"
TRAIN_FRAMES="$TRAIN_ROOT/manifest_frames"
TRAIN_BASE="$TRAIN_ROOT/region_base"
TRAIN_DEPTH="$TRAIN_ROOT/depth_proposals"
MODEL_CACHE="$ROOT/depth_layered_models"
EVAL_SOURCE="$ROOT/depth_layered_wts_vehicle16"
EVAL_ALIGNED="$ROOT/depth_layered_wts_vehicle16_aligned"
EVAL_MANIFEST="$EVAL_SOURCE/selected_manifest.jsonl"
OUT="$ROOT/supervised_depth_gate_train32_val16"
GPU_ID="${CUDA_VISIBLE_DEVICES:-3}"
GPU_ID="${GPU_ID%%,*}"

echo "=== Supervised bounded depth-residual gate: train32 -> aligned val16 ==="
echo "The model can only attenuate/amplify the frozen depth-a75 residual; geometry remains frozen."
/workspace/wan_ticft/evaluation/check_gpu_free.sh "$GPU_ID" 12000

for path in "$SOURCE_MANIFEST" "$EVAL_MANIFEST" \
  "$EVAL_ALIGNED/lossless_control/report.csv" "$EVAL_ALIGNED/depth_a75/report.csv"; do
  [[ -f "$path" ]] || { echo "Missing input: $path" >&2; exit 2; }
done

echo "[1/6] Materialize 32 independent WTS vehicle train rows"
"$PYTHON" -m wan_ticft.data.materialize_manifest_frames \
  --src_jsonl "$SOURCE_MANIFEST" \
  --out_jsonl "$TRAIN_MANIFEST" \
  --frames_root "$TRAIN_FRAMES" \
  --view_type wts_vehicle_or_ip_camera \
  --limit 32 \
  --clean

echo "[2/6] Generate correctly guarded region-medium train bases"
"$PYTHON" -m wan_ticft.evaluation.materialize_wts_distribution_gate \
  --manifest "$TRAIN_MANIFEST" \
  --out_dir "$TRAIN_BASE" \
  --views wts_vehicle_or_ip_camera \
  --limit_per_domain 32 \
  --cover_manifest_targets \
  --candidates region_norm_medium \
  --clean

echo "[3/6] Generate frozen depth-a75 train proposals"
CUDA_VISIBLE_DEVICES="$GPU_ID" \
HF_HOME="$MODEL_CACHE/hf_home" \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
"$PYTHON" -m wan_ticft.evaluation.materialize_depth_layered_renderer \
  --manifest "$TRAIN_MANIFEST" \
  --base_report "$TRAIN_BASE/region_norm_medium_report.csv" \
  --out_root "$TRAIN_DEPTH" \
  --model_cache "$MODEL_CACHE/huggingface" \
  --view_type wts_vehicle_or_ip_camera \
  --preserve_full_video \
  --limit 32 \
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
  --confidence_ema 0.65 \
  --clean

echo "[4/6] Train bounded supervised multiplier and infer aligned val16"
CUDA_VISIBLE_DEVICES="$GPU_ID" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
"$PYTHON" -m wan_ticft.evaluation.train_supervised_depth_gate \
  --train_manifest "$TRAIN_MANIFEST" \
  --train_base_report "$TRAIN_DEPTH/depth_layered_a00/report.csv" \
  --train_candidate_report "$TRAIN_DEPTH/depth_layered_a75/report.csv" \
  --eval_manifest "$EVAL_MANIFEST" \
  --eval_base_report "$EVAL_ALIGNED/lossless_control/report.csv" \
  --eval_candidate_report "$EVAL_ALIGNED/depth_a75/report.csv" \
  --out_dir "$OUT" \
  --train_limit 32 \
  --train_width 640 \
  --train_height 360 \
  --patch 128 \
  --batch_size 8 \
  --epochs 6 \
  --steps_per_epoch 100 \
  --model_width 24 \
  --lambda_edge 0.10 \
  --lambda_identity 0.01 \
  --clean

echo "[5/6] Pixel and distribution gate"
"$PYTHON" -m wan_ticft.evaluation.evaluate_video_reports \
  --manifest "$EVAL_MANIFEST" \
  --reports \
    lossless_control="$EVAL_ALIGNED/lossless_control/report.csv" \
    depth_a75="$EVAL_ALIGNED/depth_a75/report.csv" \
    supervised_gate="$OUT/report.csv" \
  --include_repeat_last \
  --offset_mode timestamp \
  --out_csv "${OUT}_eval.csv"

CUDA_VISIBLE_DEVICES="$GPU_ID" TORCH_HOME="$ROOT/tools/torch_cache" \
"$PYTHON" -m wan_ticft.evaluation.audit_perceptual_video_distribution \
  --manifest "$EVAL_MANIFEST" \
  --report lossless_control="$EVAL_ALIGNED/lossless_control/report.csv" \
  --report depth_a75="$EVAL_ALIGNED/depth_a75/report.csv" \
  --report supervised_gate="$OUT/report.csv" \
  --out_csv "${OUT}_proxy.csv" \
  --clip_frames 8 \
  --frame_stride 1

echo "[6/6] Contact sheets"
mapfile -t SAMPLE_IDS < <("$PYTHON" - <<PY
import json
for line in open("$EVAL_MANIFEST", encoding="utf-8"):
    if line.strip(): print(json.loads(line)["sample_id"])
PY
)
SHEET_ARGS=()
for sample_id in "${SAMPLE_IDS[@]}"; do SHEET_ARGS+=(--sample_id "$sample_id"); done
"$PYTHON" -m wan_ticft.evaluation.make_report_comparison_sheet \
  --manifest "$EVAL_MANIFEST" \
  --reports \
    lossless_control="$EVAL_ALIGNED/lossless_control/report.csv" \
    depth_a75="$EVAL_ALIGNED/depth_a75/report.csv" \
    supervised_gate="$OUT/report.csv" \
  "${SHEET_ARGS[@]}" \
  --out_dir "${OUT}_compare_sheets" \
  --frames 8 \
  --tile_width 220

echo
echo "eval=${OUT}_eval.csv"
echo "proxy=${OUT}_proxy.csv"
echo "history=$OUT/history.csv"
echo "sheets=${OUT}_compare_sheets"
echo "Gate: supervised_gate must beat depth_a75 on PSNR without lowering SSIM or worsening FID/FVD."

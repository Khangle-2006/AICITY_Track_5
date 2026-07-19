#!/usr/bin/env bash
set -euo pipefail

cd /workspace

PYTHON=/opt/conda/envs/wan22/bin/python
ROOT=/workspace/experiments/wan_ticft
MANIFEST="$ROOT/ltxv_bdd25_gate/dense40_manifest.jsonl"
BASE_REPORT="$ROOT/cnn_selector_perceptual_tradeoff_gate/cnn_ltxmix65/report.csv"
ORIGINAL_DEPTH="$ROOT/depth_layered_renderer_temporal_bdd24/depth_layered_a75/report.csv"
ORIGINAL_A125="$ROOT/sa2va_actor_depth_strength_extension_bdd24/actor_depth_a125/report.csv"
MASK_ROOT="$ROOT/sa2va_actor_aware_depth_bdd24"
INPAINT_ROOT="$ROOT/propainter_history_bdd3_smoke/inpaint"
DEPTH_OUT="$ROOT/propainter_clean_anchor_depth_bdd3"
ACTOR_OUT="$ROOT/propainter_clean_anchor_actor_depth_bdd3"
MODEL_CACHE="$ROOT/depth_layered_models"
GPU_ID="${CUDA_VISIBLE_DEVICES:-0}"
GPU_ID="${GPU_ID%%,*}"
SAMPLES=(
  val_bdd_pc_5k_video2556_phase2
  val_bdd_pc_5k_video8125_phase1
  val_bdd_pc_5k_video8139_phase2
)

echo "=== ProPainter clean-anchor + Sa2VA actor-depth BDD3 scored gate ==="
/workspace/wan_ticft/evaluation/check_gpu_free.sh "$GPU_ID" 12000
for path in "$MANIFEST" "$BASE_REPORT" "$ORIGINAL_DEPTH" "$ORIGINAL_A125"; do
  [[ -s "$path" ]] || { echo "Missing input: $path" >&2; exit 2; }
done
for sample_id in "${SAMPLES[@]}"; do
  [[ -s "$INPAINT_ROOT/$sample_id/frames/0003.png" ]] || {
    echo "Missing ProPainter anchor: $sample_id" >&2
    exit 2
  }
done

SAMPLE_ARGS=()
for sample_id in "${SAMPLES[@]}"; do
  SAMPLE_ARGS+=(--sample_id "$sample_id")
done

echo "[1/5] Render depth background from ProPainter-cleaned anchors"
CUDA_VISIBLE_DEVICES="$GPU_ID" \
HF_HOME="$MODEL_CACHE/hf_home" \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
"$PYTHON" -m wan_ticft.evaluation.materialize_depth_layered_renderer \
  --manifest "$MANIFEST" \
  --base_report "$BASE_REPORT" \
  --out_root "$DEPTH_OUT" \
  --model_cache "$MODEL_CACHE/huggingface" \
  --anchor_override_root "$INPAINT_ROOT" \
  "${SAMPLE_ARGS[@]}" \
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

echo "[2/5] Apply cached Sa2VA masks at validated a125 strength"
"$PYTHON" -m wan_ticft.evaluation.materialize_actor_depth_strength_sweep \
  --manifest "$DEPTH_OUT/selected_manifest.jsonl" \
  --base_report "$BASE_REPORT" \
  --depth_report "$DEPTH_OUT/depth_layered_a75/report.csv" \
  --mask_dir "$MASK_ROOT" \
  --out_dir "$ACTOR_OUT" \
  --strength propainter_actor_depth_a125:1.6666666667 \
  --clean

echo "[3/5] Pixel metrics"
"$PYTHON" -m wan_ticft.evaluation.evaluate_video_reports \
  --manifest "$DEPTH_OUT/selected_manifest.jsonl" \
  --reports \
    cnn_ltxmix65="$BASE_REPORT" \
    depth_a75="$ORIGINAL_DEPTH" \
    actor_depth_a125="$ORIGINAL_A125" \
    propainter_depth_a75="$DEPTH_OUT/depth_layered_a75/report.csv" \
    propainter_actor_depth_a125="$ACTOR_OUT/propainter_actor_depth_a125/report.csv" \
  --include_repeat_last \
  --offset_mode timestamp \
  --out_csv "${ACTOR_OUT}_eval.csv"

echo "[4/5] FID/FVD proxy"
CUDA_VISIBLE_DEVICES="$GPU_ID" TORCH_HOME="$ROOT/tools/torch_cache" \
"$PYTHON" -m wan_ticft.evaluation.audit_perceptual_video_distribution \
  --manifest "$DEPTH_OUT/selected_manifest.jsonl" \
  --report cnn_ltxmix65="$BASE_REPORT" \
  --report depth_a75="$ORIGINAL_DEPTH" \
  --report actor_depth_a125="$ORIGINAL_A125" \
  --report propainter_depth_a75="$DEPTH_OUT/depth_layered_a75/report.csv" \
  --report propainter_actor_depth_a125="$ACTOR_OUT/propainter_actor_depth_a125/report.csv" \
  --out_csv "${ACTOR_OUT}_proxy.csv" \
  --clip_frames 16 \
  --frame_stride 4

echo "[5/5] Contact sheets"
"$PYTHON" -m wan_ticft.evaluation.make_report_comparison_sheet \
  --manifest "$DEPTH_OUT/selected_manifest.jsonl" \
  --reports \
    actor_depth_a125="$ORIGINAL_A125" \
    propainter_depth_a75="$DEPTH_OUT/depth_layered_a75/report.csv" \
    propainter_actor_depth_a125="$ACTOR_OUT/propainter_actor_depth_a125/report.csv" \
  "${SAMPLE_ARGS[@]}" \
  --out_dir "${ACTOR_OUT}_compare_sheets" \
  --frames 8 \
  --tile_width 260

echo
echo "eval=${ACTOR_OUT}_eval.csv"
echo "proxy=${ACTOR_OUT}_proxy.csv"
echo "sheets=${ACTOR_OUT}_compare_sheets"
echo "Reject immediately unless the clean-anchor actor a125 beats original actor a125 on both pixel and proxy metrics."

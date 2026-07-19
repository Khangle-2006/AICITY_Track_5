#!/bin/bash
set -euo pipefail
cd /workspace
export CPU_THREADS="${CPU_THREADS:-4}"
export OMP_NUM_THREADS=$CPU_THREADS OPENBLAS_NUM_THREADS=$CPU_THREADS MKL_NUM_THREADS=$CPU_THREADS NUMEXPR_NUM_THREADS=$CPU_THREADS VECLIB_MAXIMUM_THREADS=$CPU_THREADS
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2}"

PYTHON=${PYTHON:-/opt/conda/envs/wan22/bin/python}
ROOT=/workspace/experiments/wan_ticft
MANIFEST=$ROOT/ltxv_bdd25_gate/selected_manifest.jsonl
MODEL_CACHE=$ROOT/depth_layered_models/huggingface
REAL=$ROOT/frames_balanced_val_125/val
A75DIR=$ROOT/bdd_supervised_ltx_depth_eval24/depth/depth_layered_a75
BASE=${BASE:-$ROOT/basemode_gate/mf_a75_h4_video}
OUT=$ROOT/actorflow_relaxed_sweep_gate

rm -rf "$OUT"

run_variant() {
  local tag="$1"
  shift
  echo "=== render $tag ==="
  "$PYTHON" -m wan_ticft.evaluation.materialize_actor_flow_deghost \
    --manifest "$MANIFEST" \
    --base_report "$BASE/report.csv" \
    --out_root "$OUT" \
    --model_cache "$MODEL_CACHE" \
    --view_type bdd \
    --limit 24 \
    --tag "$tag" \
    --flow_scale 0.5 \
    --segment_batch 4 \
    "$@"
}

run_variant mf_actorflow_relax_dyn \
  --dynamic_only \
  --actor_strength 0.70 \
  --mask_dilate 13 \
  --mask_blur 4.0 \
  --dynamic_residual_thresh 2.4 \
  --dynamic_min_area 70 \
  --dynamic_max_area_frac 0.08 \
  --dynamic_max_bbox_height_frac 0.45 \
  --dynamic_exclude_bottom_frac 0.0 \
  --component_validate \
  --component_min_area 70 \
  --component_max_area_frac 0.07 \
  --component_max_bbox_height_frac 0.42 \
  --component_exclude_bottom_frac 0.0 \
  --component_boundary 13 \
  --component_max_boundary_diff 38 \
  --component_min_valid_frac 0.84

run_variant mf_actorflow_relax_dyn_strong \
  --dynamic_only \
  --actor_strength 0.85 \
  --mask_dilate 15 \
  --mask_blur 4.5 \
  --dynamic_residual_thresh 2.2 \
  --dynamic_min_area 60 \
  --dynamic_max_area_frac 0.10 \
  --dynamic_max_bbox_height_frac 0.50 \
  --dynamic_exclude_bottom_frac 0.0 \
  --component_validate \
  --component_min_area 60 \
  --component_max_area_frac 0.09 \
  --component_max_bbox_height_frac 0.48 \
  --component_exclude_bottom_frac 0.0 \
  --component_boundary 11 \
  --component_max_boundary_diff 45 \
  --component_min_valid_frac 0.80

run_variant mf_actorflow_semantic_valid \
  --actor_strength 0.65 \
  --mask_dilate 13 \
  --mask_blur 4.0 \
  --component_validate \
  --component_min_area 80 \
  --component_max_area_frac 0.055 \
  --component_max_bbox_height_frac 0.42 \
  --component_exclude_bottom_frac 0.0 \
  --component_boundary 15 \
  --component_max_boundary_diff 34 \
  --component_min_valid_frac 0.88

echo "=== PIXEL ==="
"$PYTHON" -m wan_ticft.evaluation.evaluate_video_reports \
  --manifest "$MANIFEST" \
  --reports \
    mf_video="$BASE/report.csv" \
    relax_dyn="$OUT/mf_actorflow_relax_dyn/report.csv" \
    relax_dyn_strong="$OUT/mf_actorflow_relax_dyn_strong/report.csv" \
    semantic_valid="$OUT/mf_actorflow_semantic_valid/report.csv" \
  --offset_mode timestamp \
  --out_csv "$OUT/eval_pixel.csv"

echo "=== REAL-FID ==="
"$PYTHON" -m wan_ticft.evaluation.compute_real_fid \
  --real_root "$REAL" \
  --work "$OUT/_fid" \
  --limit 24 \
  --candidate "a75_single=$A75DIR/videos" \
  --candidate "mf_video=$BASE/videos" \
  --candidate "relax_dyn=$OUT/mf_actorflow_relax_dyn/videos" \
  --candidate "relax_dyn_strong=$OUT/mf_actorflow_relax_dyn_strong/videos" \
  --candidate "semantic_valid=$OUT/mf_actorflow_semantic_valid/videos"

echo "output=$OUT"

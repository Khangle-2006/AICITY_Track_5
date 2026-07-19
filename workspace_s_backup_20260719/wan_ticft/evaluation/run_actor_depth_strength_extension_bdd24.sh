#!/usr/bin/env bash
set -euo pipefail

cd /workspace

PYTHON=/opt/conda/envs/wan22/bin/python
ROOT=/workspace/experiments/wan_ticft
SOURCE="$ROOT/sa2va_actor_aware_depth_bdd24"
BASE_REPORT="$ROOT/cnn_selector_perceptual_tradeoff_gate/cnn_ltxmix65/report.csv"
DEPTH_REPORT="$ROOT/depth_layered_renderer_temporal_bdd24/depth_layered_a75/report.csv"
SWEEP1="$ROOT/sa2va_actor_depth_strength_sweep_bdd24"
OUT="$ROOT/sa2va_actor_depth_strength_extension_bdd24"
GPU_ID="${CUDA_VISIBLE_DEVICES:-0}"
GPU_ID="${GPU_ID%%,*}"

echo "=== Cached-mask actor-depth extension: a125/a130/a140/a150 ==="
echo "Materialization is CPU-only; GPU $GPU_ID is used only for FID/FVD."
for path in \
  "$SOURCE/selected_manifest.jsonl" \
  "$BASE_REPORT" \
  "$DEPTH_REPORT" \
  "$SWEEP1/actor_depth_a120/report.csv"; do
  [[ -s "$path" ]] || { echo "Missing input: $path" >&2; exit 2; }
done

"$PYTHON" -m wan_ticft.evaluation.materialize_actor_depth_strength_sweep \
  --manifest "$SOURCE/selected_manifest.jsonl" \
  --base_report "$BASE_REPORT" \
  --depth_report "$DEPTH_REPORT" \
  --mask_dir "$SOURCE" \
  --out_dir "$OUT" \
  --strength actor_depth_a125:1.6666666667 \
  --strength actor_depth_a130:1.7333333333 \
  --strength actor_depth_a140:1.8666666667 \
  --strength actor_depth_a150:2.00 \
  --clean

"$PYTHON" -m wan_ticft.evaluation.evaluate_video_reports \
  --manifest "$SOURCE/selected_manifest.jsonl" \
  --reports \
    actor_depth_a100="$SOURCE/actor_depth_a100/report.csv" \
    actor_depth_a120="$SWEEP1/actor_depth_a120/report.csv" \
    actor_depth_a125="$OUT/actor_depth_a125/report.csv" \
    actor_depth_a130="$OUT/actor_depth_a130/report.csv" \
    actor_depth_a140="$OUT/actor_depth_a140/report.csv" \
    actor_depth_a150="$OUT/actor_depth_a150/report.csv" \
  --include_repeat_last \
  --offset_mode timestamp \
  --out_csv "${OUT}_eval.csv"

/workspace/wan_ticft/evaluation/check_gpu_free.sh "$GPU_ID" 7000
CUDA_VISIBLE_DEVICES="$GPU_ID" TORCH_HOME="$ROOT/tools/torch_cache" \
"$PYTHON" -m wan_ticft.evaluation.audit_perceptual_video_distribution \
  --manifest "$SOURCE/selected_manifest.jsonl" \
  --report actor_depth_a100="$SOURCE/actor_depth_a100/report.csv" \
  --report actor_depth_a120="$SWEEP1/actor_depth_a120/report.csv" \
  --report actor_depth_a125="$OUT/actor_depth_a125/report.csv" \
  --report actor_depth_a130="$OUT/actor_depth_a130/report.csv" \
  --report actor_depth_a140="$OUT/actor_depth_a140/report.csv" \
  --report actor_depth_a150="$OUT/actor_depth_a150/report.csv" \
  --out_csv "${OUT}_proxy.csv" \
  --clip_frames 16 \
  --frame_stride 4

"$PYTHON" -m wan_ticft.evaluation.make_report_comparison_sheet \
  --manifest "$SOURCE/selected_manifest.jsonl" \
  --reports \
    actor_depth_a120="$SWEEP1/actor_depth_a120/report.csv" \
    actor_depth_a125="$OUT/actor_depth_a125/report.csv" \
    actor_depth_a130="$OUT/actor_depth_a130/report.csv" \
    actor_depth_a140="$OUT/actor_depth_a140/report.csv" \
    actor_depth_a150="$OUT/actor_depth_a150/report.csv" \
  --sample_id val_bdd_pc_5k_video7040_phase0 \
  --sample_id val_bdd_pc_5k_video6075_phase3 \
  --sample_id val_bdd_pc_5k_video8139_phase2 \
  --sample_id val_bdd_pc_5k_video3207_phase4 \
  --out_dir "${OUT}_compare_sheets" \
  --frames 8 \
  --tile_width 220

echo
echo "eval=${OUT}_eval.csv"
echo "proxy=${OUT}_proxy.csv"
echo "sheets=${OUT}_compare_sheets"
echo "Stop at the first strength where aggregate pixel metrics or both proxies turn worse."

#!/usr/bin/env bash
set -euo pipefail

cd /workspace

PYTHON=/opt/conda/envs/wan22/bin/python
ROOT=/workspace/experiments/wan_ticft
SOURCE="$ROOT/sa2va_actor_aware_depth_bdd24"
BASE_REPORT="$ROOT/cnn_selector_perceptual_tradeoff_gate/cnn_ltxmix65/report.csv"
DEPTH_REPORT="$ROOT/depth_layered_renderer_temporal_bdd24/depth_layered_a75/report.csv"
OUT="$ROOT/sa2va_actor_depth_strength_sweep_bdd24"
GPU_ID="${CUDA_VISIBLE_DEVICES:-3}"
GPU_ID="${GPU_ID%%,*}"

echo "=== Cached-mask actor-depth strength sweep BDD24 ==="
echo "Materialization is CPU-only; GPU $GPU_ID is used only for FID/FVD features."
for path in "$SOURCE/selected_manifest.jsonl" "$BASE_REPORT" "$DEPTH_REPORT"; do
  [[ -s "$path" ]] || { echo "Missing input: $path" >&2; exit 2; }
done

"$PYTHON" -m wan_ticft.evaluation.materialize_actor_depth_strength_sweep \
  --manifest "$SOURCE/selected_manifest.jsonl" \
  --base_report "$BASE_REPORT" \
  --depth_report "$DEPTH_REPORT" \
  --mask_dir "$SOURCE" \
  --out_dir "$OUT" \
  --strength actor_depth_a105:1.40 \
  --strength actor_depth_a110:1.4666666667 \
  --strength actor_depth_a120:1.60 \
  --clean

"$PYTHON" -m wan_ticft.evaluation.evaluate_video_reports \
  --manifest "$SOURCE/selected_manifest.jsonl" \
  --reports \
    depth_a75="$DEPTH_REPORT" \
    actor_depth_a100="$SOURCE/actor_depth_a100/report.csv" \
    actor_depth_a105="$OUT/actor_depth_a105/report.csv" \
    actor_depth_a110="$OUT/actor_depth_a110/report.csv" \
    actor_depth_a120="$OUT/actor_depth_a120/report.csv" \
  --include_repeat_last \
  --offset_mode timestamp \
  --out_csv "${OUT}_eval.csv"

/workspace/wan_ticft/evaluation/check_gpu_free.sh "$GPU_ID" 7000
CUDA_VISIBLE_DEVICES="$GPU_ID" TORCH_HOME="$ROOT/tools/torch_cache" \
"$PYTHON" -m wan_ticft.evaluation.audit_perceptual_video_distribution \
  --manifest "$SOURCE/selected_manifest.jsonl" \
  --report depth_a75="$DEPTH_REPORT" \
  --report actor_depth_a100="$SOURCE/actor_depth_a100/report.csv" \
  --report actor_depth_a105="$OUT/actor_depth_a105/report.csv" \
  --report actor_depth_a110="$OUT/actor_depth_a110/report.csv" \
  --report actor_depth_a120="$OUT/actor_depth_a120/report.csv" \
  --out_csv "${OUT}_proxy.csv" \
  --clip_frames 16 \
  --frame_stride 4

"$PYTHON" -m wan_ticft.evaluation.make_report_comparison_sheet \
  --manifest "$SOURCE/selected_manifest.jsonl" \
  --reports \
    actor_depth_a100="$SOURCE/actor_depth_a100/report.csv" \
    actor_depth_a105="$OUT/actor_depth_a105/report.csv" \
    actor_depth_a110="$OUT/actor_depth_a110/report.csv" \
    actor_depth_a120="$OUT/actor_depth_a120/report.csv" \
  --sample_id val_bdd_pc_5k_video7040_phase0 \
  --sample_id val_bdd_pc_5k_video6075_phase3 \
  --sample_id val_bdd_pc_5k_video8139_phase2 \
  --sample_id val_bdd_pc_5k_video3207_phase4 \
  --out_dir "${OUT}_compare_sheets" \
  --frames 8 \
  --tile_width 240

echo
echo "eval=${OUT}_eval.csv"
echo "proxy=${OUT}_proxy.csv"
echo "sheets=${OUT}_compare_sheets"

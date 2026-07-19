#!/usr/bin/env bash
set -euo pipefail

cd /workspace

PYTHON=/opt/conda/envs/wan22/bin/python
ROOT=/workspace/experiments/wan_ticft
MANIFEST="$ROOT/ltxv_bdd25_gate/dense40_manifest.jsonl"
BASE_REPORT="$ROOT/cnn_selector_perceptual_tradeoff_gate/cnn_ltxmix65/report.csv"
DEPTH_REPORT="$ROOT/depth_layered_renderer_temporal_bdd24/depth_layered_a75/report.csv"
OUT="$ROOT/sa2va_actor_aware_depth_gate3"
MODEL="${SA2VA_MODEL:-ByteDance/Sa2VA-4B}"
REVISION="${SA2VA_REVISION:-3fee777d49ee9276eac51ea3e5f9b69e81d09be6}"
GPU_ID="${CUDA_VISIBLE_DEVICES:-2}"
GPU_ID="${GPU_ID%%,*}"
CHUNK_FRAMES="${SA2VA_CHUNK_FRAMES:-3}"

echo "=== Sa2VA actor-aware depth background gate3 ==="
echo "Frozen actor pixels from cnn_ltxmix65; stronger depth texture only outside tracked pedestrian."
echo "chunk_frames=$CHUNK_FRAMES"
/workspace/wan_ticft/evaluation/check_gpu_free.sh "$GPU_ID" 22000
for path in "$MANIFEST" "$BASE_REPORT" "$DEPTH_REPORT"; do
  [[ -s "$path" ]] || { echo "Missing input: $path" >&2; exit 2; }
done

rm -rf "$OUT"
CUDA_VISIBLE_DEVICES="$GPU_ID" \
HF_HOME=/root/.cache/huggingface \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
TOKENIZERS_PARALLELISM=false \
"$PYTHON" -m wan_ticft.evaluation.run_sa2va_actor_aware_depth_gate \
  --manifest "$MANIFEST" \
  --base_report "$BASE_REPORT" \
  --depth_report "$DEPTH_REPORT" \
  --out_dir "$OUT" \
  --model "$MODEL" \
  --revision "$REVISION" \
  --sample_id val_bdd_pc_5k_video2556_phase2 \
  --sample_id val_bdd_pc_5k_video8125_phase1 \
  --sample_id val_bdd_pc_5k_video8139_phase2 \
  --chunk_frames "$CHUNK_FRAMES"

echo "[score] Pixel metrics"
"$PYTHON" -m wan_ticft.evaluation.evaluate_video_reports \
  --manifest "$OUT/selected_manifest.jsonl" \
  --reports \
    cnn_ltxmix65="$BASE_REPORT" \
    depth_a75="$DEPTH_REPORT" \
    actor_depth_a75="$OUT/actor_depth_a75/report.csv" \
    actor_depth_a90="$OUT/actor_depth_a90/report.csv" \
    actor_depth_a100="$OUT/actor_depth_a100/report.csv" \
  --include_repeat_last \
  --offset_mode timestamp \
  --out_csv "${OUT}_eval.csv"

echo "[score] FID/FVD proxy"
CUDA_VISIBLE_DEVICES="$GPU_ID" TORCH_HOME="$ROOT/tools/torch_cache" \
"$PYTHON" -m wan_ticft.evaluation.audit_perceptual_video_distribution \
  --manifest "$OUT/selected_manifest.jsonl" \
  --report cnn_ltxmix65="$BASE_REPORT" \
  --report depth_a75="$DEPTH_REPORT" \
  --report actor_depth_a75="$OUT/actor_depth_a75/report.csv" \
  --report actor_depth_a90="$OUT/actor_depth_a90/report.csv" \
  --report actor_depth_a100="$OUT/actor_depth_a100/report.csv" \
  --out_csv "${OUT}_proxy.csv" \
  --clip_frames 16 \
  --frame_stride 4

"$PYTHON" -m wan_ticft.evaluation.make_report_comparison_sheet \
  --manifest "$OUT/selected_manifest.jsonl" \
  --reports \
    cnn_ltxmix65="$BASE_REPORT" \
    depth_a75="$DEPTH_REPORT" \
    actor_depth_a90="$OUT/actor_depth_a90/report.csv" \
    actor_depth_a100="$OUT/actor_depth_a100/report.csv" \
  --sample_id val_bdd_pc_5k_video2556_phase2 \
  --sample_id val_bdd_pc_5k_video8125_phase1 \
  --sample_id val_bdd_pc_5k_video8139_phase2 \
  --out_dir "${OUT}_compare_sheets" \
  --frames 8 \
  --tile_width 260

echo
echo "eval=${OUT}_eval.csv"
echo "proxy=${OUT}_proxy.csv"
echo "sheets=${OUT}_compare_sheets"
echo "diagnostics=$OUT/diagnostics.csv"
echo
echo "Reject if actor edges change, a duplicate actor is selected, or FVD worsens versus depth_a75."
echo "Scale to BDD24 only if a90/a100 improves both pixel quality and FID/FVD proxy."

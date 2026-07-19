#!/usr/bin/env bash
set -euo pipefail

cd /workspace

ROOT=/workspace/experiments/wan_ticft
MANIFEST="$ROOT/skyreels_v2_df_v2v_bdd1_smoke/selected_manifest.jsonl"
BASE_REPORT="$ROOT/cnn_selector_perceptual_tradeoff_gate/cnn_ltxmix65/report.csv"
SKY_REPORT="$ROOT/skyreels_v2_df_v2v_bdd1_smoke/report.csv"
OUT="$ROOT/motion_locked_texture_skyreels_bdd1_lossless"
GPU_ID="${CUDA_VISIBLE_DEVICES:-3}"
GPU_ID="${GPU_ID%%,*}"

echo "=== Motion-locked SkyReels texture BDD1 smoke ==="
echo "This script reuses existing CNN and SkyReels videos; it does not run SkyReels inference."
echo "manifest=$MANIFEST"
echo "out=$OUT"

for path in "$MANIFEST" "$BASE_REPORT" "$SKY_REPORT"; do
  if [[ ! -f "$path" ]]; then
    echo "Missing input: $path" >&2
    exit 2
  fi
done

/opt/conda/envs/wan22/bin/python -m wan_ticft.evaluation.materialize_motion_locked_texture \
  --manifest "$MANIFEST" \
  --base_report "$BASE_REPORT" \
  --skyreels_report "$SKY_REPORT" \
  --out_root "$OUT" \
  --limit 1 \
  --clean

/opt/conda/envs/wan22/bin/python -m wan_ticft.evaluation.evaluate_video_reports \
  --manifest "$OUT/selected_manifest.jsonl" \
  --reports \
    cnn_ltxmix65="$BASE_REPORT" \
    lossless_control="$OUT/lossless_control/report.csv" \
    sky_yhf_a15="$OUT/sky_yhf_a15/report.csv" \
    sky_yhf_a25="$OUT/sky_yhf_a25/report.csv" \
    sky_yhf_a35="$OUT/sky_yhf_a35/report.csv" \
    condition_yhf_a25="$OUT/condition_yhf_a25/report.csv" \
  --include_repeat_last \
  --offset_mode timestamp \
  --out_csv "${OUT}_eval.csv"

CUDA_VISIBLE_DEVICES="$GPU_ID" \
TORCH_HOME="$ROOT/tools/torch_cache" \
/opt/conda/envs/wan22/bin/python -m wan_ticft.evaluation.audit_perceptual_video_distribution \
  --manifest "$OUT/selected_manifest.jsonl" \
  --report cnn_ltxmix65="$BASE_REPORT" \
  --report lossless_control="$OUT/lossless_control/report.csv" \
  --report sky_yhf_a15="$OUT/sky_yhf_a15/report.csv" \
  --report sky_yhf_a25="$OUT/sky_yhf_a25/report.csv" \
  --report sky_yhf_a35="$OUT/sky_yhf_a35/report.csv" \
  --report condition_yhf_a25="$OUT/condition_yhf_a25/report.csv" \
  --out_csv "${OUT}_proxy.csv" \
  --clip_frames 16 \
  --frame_stride 2

/opt/conda/envs/wan22/bin/python -m wan_ticft.evaluation.make_report_comparison_sheet \
  --manifest "$OUT/selected_manifest.jsonl" \
  --reports \
    cnn_ltxmix65="$BASE_REPORT" \
    lossless_control="$OUT/lossless_control/report.csv" \
    sky_yhf_a15="$OUT/sky_yhf_a15/report.csv" \
    sky_yhf_a25="$OUT/sky_yhf_a25/report.csv" \
    sky_yhf_a35="$OUT/sky_yhf_a35/report.csv" \
    condition_yhf_a25="$OUT/condition_yhf_a25/report.csv" \
  --sample_id val_bdd_pc_5k_video7040_phase0 \
  --out_dir "${OUT}_compare_sheets" \
  --frames 8 \
  --tile_width 220

echo
echo "Smoke outputs:"
echo "  eval=${OUT}_eval.csv"
echo "  proxy=${OUT}_proxy.csv"
echo "  sheets=${OUT}_compare_sheets"
echo
echo "Promotion rule:"
echo "  - Verify lossless_control matches cnn_ltxmix65 within 0.01 PSNR and 0.0002 SSIM."
echo "  - If it does not, stop: the gate still has an I/O/alignment confound."
echo "  - Do not scale from one sample alone."
echo "  - Continue only if a SkyReels variant improves FID and FVD proxy versus lossless_control,"
echo "    does not lose more than 0.10 PSNR or 0.002 SSIM, and the sheet shows sharper"
echo "    background without halos, doubled edges, or temporal texture crawl."
echo "  - If condition_yhf wins but SkyReels does not, pursue reference-guided restoration"
echo "    rather than SkyReels texture transfer."

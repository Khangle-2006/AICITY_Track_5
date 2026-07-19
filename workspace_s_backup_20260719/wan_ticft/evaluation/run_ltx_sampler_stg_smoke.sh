#!/usr/bin/env bash
set -euo pipefail

cd /workspace

PYTHON=${PYTHON:-/opt/conda/envs/wan22/bin/python}
ROOT=${ROOT:-/workspace/experiments/wan_ticft}
MANIFEST=${MANIFEST:-$ROOT/manifests/val_ticft_balanced_125_frames.jsonl}
OUT=${OUT:-$ROOT/ltx_sampler_stg_smoke4}
DENSE_MANIFEST="$OUT/dense40_manifest.jsonl"
PIXEL_CSV=${PIXEL_CSV:-$ROOT/ltx_sampler_stg_smoke4_eval.csv}
PROXY_CSV=${PROXY_CSV:-$ROOT/ltx_sampler_stg_smoke4_proxy.csv}
BASE_LTX_REPORT=${BASE_LTX_REPORT:-$ROOT/ltxv_bdd25_gate/report.csv}
CNN65_REPORT=${CNN65_REPORT:-$ROOT/cnn_selector_perceptual_tradeoff_gate/cnn_ltxmix65/report.csv}

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-3}"
GPU_ID="${CUDA_VISIBLE_DEVICES%%,*}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

echo "=== LTX sampler/STG smoke: model-level anti-ghosting route ==="
echo "This tests sampler behavior, not a postprocess/mix patch."
/workspace/wan_ticft/evaluation/check_gpu_free.sh "$GPU_ID" 24000

SAMPLES=(
  val_bdd_pc_5k_video7040_phase0
  val_bdd_pc_5k_video2149_phase0
  val_bdd_pc_5k_video1407_phase0
  val_bdd_pc_5k_video3237_phase0
)

run_variant() {
  local name="$1"
  local config="$2"
  local out_dir="$OUT/$name"
  echo
  echo "[generate] $name config=$config"
  "$PYTHON" -m wan_ticft.evaluation.run_ltxv_val_gate \
    --manifest "$MANIFEST" \
    --out_dir "$out_dir" \
    --ltx_repo /workspace/third_party/LTX-Video \
    --ltx_python "$ROOT/ltxv_env/bin/python" \
    --pipeline_config "$config" \
    --sample_ids "${SAMPLES[@]}" \
    --max_history_frames 41 \
    --limit 4 \
    --resume
}

run_variant stg_attention_s020 /workspace/wan_ticft/evaluation/ltxv_2b_stg_attention_s020.yaml
run_variant stg_residual_s020 /workspace/wan_ticft/evaluation/ltxv_2b_stg_residual_s020.yaml
run_variant deterministic_decode /workspace/wan_ticft/evaluation/ltxv_2b_deterministic_decode.yaml

rm -f "$DENSE_MANIFEST"
rm -rf "$OUT/dense40_frames"
"$PYTHON" -m wan_ticft.data.materialize_future_manifest \
  --src_jsonl "$OUT/stg_attention_s020/selected_manifest.jsonl" \
  --out_jsonl "$DENSE_MANIFEST" \
  --frames_root "$OUT/dense40_frames" \
  --future_frames 40

echo
echo "=== Pixel metrics ==="
"$PYTHON" -m wan_ticft.evaluation.evaluate_video_reports \
  --manifest "$DENSE_MANIFEST" \
  --reports \
    ltx_base="$BASE_LTX_REPORT" \
    cnn_ltxmix65="$CNN65_REPORT" \
    stg_attention_s020="$OUT/stg_attention_s020/report.csv" \
    stg_residual_s020="$OUT/stg_residual_s020/report.csv" \
    deterministic_decode="$OUT/deterministic_decode/report.csv" \
  --include_repeat_last \
  --offset_mode timestamp \
  --out_csv "$PIXEL_CSV"

echo
echo "=== Relative FID/FVD proxy ==="
/workspace/wan_ticft/evaluation/check_gpu_free.sh "$GPU_ID" 3000
CUDA_VISIBLE_DEVICES="$GPU_ID" \
TORCH_HOME="$ROOT/tools/torch_cache" \
"$PYTHON" -m wan_ticft.evaluation.audit_perceptual_video_distribution \
  --manifest "$DENSE_MANIFEST" \
  --report "ltx_base=$BASE_LTX_REPORT" \
  --report "cnn_ltxmix65=$CNN65_REPORT" \
  --report "stg_attention_s020=$OUT/stg_attention_s020/report.csv" \
  --report "stg_residual_s020=$OUT/stg_residual_s020/report.csv" \
  --report "deterministic_decode=$OUT/deterministic_decode/report.csv" \
  --out_csv "$PROXY_CSV"

echo
echo "=== Contact sheets ==="
for name in stg_attention_s020 stg_residual_s020 deterministic_decode; do
  "$PYTHON" -m baseline.video_contact_sheet \
    --input "$OUT/$name/videos" \
    --mode videos \
    --out_dir "${OUT}_${name}_sheets" \
    --samples 16 \
    --tile_width 320 \
    --cols 4
done

echo
echo "Outputs:"
echo "  videos: $OUT"
echo "  pixel: $PIXEL_CSV"
echo "  proxy: $PROXY_CSV"
echo "  sheets: ${OUT}_*_sheets"
echo
echo "Gate rule:"
echo "  - Continue only if an STG/decode variant improves FVD/FID proxy versus ltx_base"
echo "    or visibly reduces early double-exposure without losing >0.20 PSNR."
echo "  - If all variants worsen proxy and still show ghost/blur, close LTX sampler tuning."

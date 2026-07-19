#!/usr/bin/env bash
set -euo pipefail

cd /workspace

PYTHON=${PYTHON:-/opt/conda/envs/wan22/bin/python}
ROOT=${ROOT:-/workspace/experiments/wan_ticft}
OUT=${OUT:-$ROOT/ltxv_bdd_minimal_prompt_smoke4}
LOWFREQ_OUT=${LOWFREQ_OUT:-$ROOT/ltxv_bdd_minimal_prompt_lowfreq_s16_a50_smoke4}
MANIFEST=${MANIFEST:-$ROOT/manifests/val_ticft_balanced_125_frames.jsonl}
DENSE_MANIFEST=${DENSE_MANIFEST:-$OUT/dense40_manifest.jsonl}
LEGACY_REPORT=${LEGACY_REPORT:-$ROOT/ltxv_bdd25_gate/report.csv}
TRANSITION16_REPORT=${TRANSITION16_REPORT:-$ROOT/ltxv_bdd25_transitions/ltx_transition_16/report.csv}
GPU_ID="${CUDA_VISIBLE_DEVICES:-3}"
GPU_ID="${GPU_ID%%,*}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-3}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

echo "=== LTX BDD minimal prompt + low-frequency correction smoke: four known prompt-sensitive samples ==="
/workspace/wan_ticft/evaluation/check_gpu_free.sh "$GPU_ID" 24000

"$PYTHON" -m wan_ticft.evaluation.run_ltxv_val_gate \
  --manifest "$MANIFEST" \
  --out_dir "$OUT" \
  --ltx_repo /workspace/third_party/LTX-Video \
  --ltx_python "$ROOT/ltxv_env/bin/python" \
  --pipeline_config /workspace/wan_ticft/evaluation/ltxv_2b_gate.yaml \
  --sample_ids \
    val_bdd_pc_5k_video4166_phase3 \
    val_bdd_pc_5k_video6180_phase0 \
    val_bdd_pc_5k_video3207_phase4 \
    val_bdd_pc_5k_video5447_phase3 \
  --max_history_frames 41 \
  --prompt_mode minimal \
  --limit 4 \
  --resume

rm -f "$DENSE_MANIFEST"
rm -rf "$OUT/dense40_frames"
"$PYTHON" -m wan_ticft.data.materialize_future_manifest \
  --src_jsonl "$OUT/selected_manifest.jsonl" \
  --out_jsonl "$DENSE_MANIFEST" \
  --frames_root "$OUT/dense40_frames" \
  --future_frames 40

echo
echo "=== Materialize low-frequency correction over minimal prompt outputs ==="
"$PYTHON" -m wan_ticft.evaluation.materialize_ltx_region_lowfreq \
  --manifest "$DENSE_MANIFEST" \
  --ltx_report "$OUT/report.csv" \
  --out_dir "$LOWFREQ_OUT" \
  --sigma 16 \
  --alpha 0.50 \
  --clean

echo
echo "=== Pixel metrics on the same four samples ==="
"$PYTHON" -m wan_ticft.evaluation.evaluate_video_reports \
  --manifest "$DENSE_MANIFEST" \
  --reports \
    minimal="$OUT/report.csv" \
    minimal_lowfreq="$LOWFREQ_OUT/report.csv" \
    legacy_blank="$LEGACY_REPORT" \
    transition16="$TRANSITION16_REPORT" \
  --offset_mode timestamp \
  --out_csv "${OUT}_minimal_lowfreq_dense40_eval.csv"

echo
echo "=== Relative FID/FVD proxy on the same four samples ==="
/workspace/wan_ticft/evaluation/check_gpu_free.sh "$GPU_ID" 3000
CUDA_VISIBLE_DEVICES="$GPU_ID" \
TORCH_HOME="$ROOT/tools/torch_cache" \
"$PYTHON" -m wan_ticft.evaluation.audit_perceptual_video_distribution \
  --manifest "$DENSE_MANIFEST" \
  --report "minimal=$OUT/report.csv" \
  --report "minimal_lowfreq=$LOWFREQ_OUT/report.csv" \
  --report "legacy_blank=$LEGACY_REPORT" \
  --report "transition16=$TRANSITION16_REPORT" \
  --out_csv "${OUT}_minimal_lowfreq_perceptual_proxy.csv"

echo
echo "=== Contact sheets ==="
"$PYTHON" -m baseline.video_contact_sheet \
  --input "$OUT/videos" \
  --mode videos \
  --out_dir "${OUT}_sheets" \
  --samples 16 \
  --tile_width 320 \
  --cols 4
"$PYTHON" -m baseline.video_contact_sheet \
  --input "$LOWFREQ_OUT/videos" \
  --mode videos \
  --out_dir "${LOWFREQ_OUT}_sheets" \
  --samples 16 \
  --tile_width 320 \
  --cols 4

echo
echo "Gate rule:"
echo "  - Minimal prompt must improve transition16 or legacy_blank on FID/FVD proxy"
echo "    without any sample losing >0.50 PSNR versus the same baseline."
echo "  - Minimal+lowfreq must improve both FID and FVD proxy versus minimal and"
echo "    transition16; pixel gains alone are not enough."
echo "  - If neither passes, retain current LTX transition16 prompt behavior and"
echo "    close prompt/lowfreq combination work."

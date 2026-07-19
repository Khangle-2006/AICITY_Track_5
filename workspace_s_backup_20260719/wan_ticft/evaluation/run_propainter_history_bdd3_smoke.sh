#!/usr/bin/env bash
set -euo pipefail

cd /workspace

PYTHON=/opt/conda/envs/wan22/bin/python
ROOT=/workspace/experiments/wan_ticft
REPO=/workspace/third_party/ProPainter
ENV="$ROOT/propainter_env"
OUT="$ROOT/propainter_history_bdd3_smoke"
MANIFEST="$ROOT/ltxv_bdd25_gate/dense40_manifest.jsonl"
MASK_ROOT="$ROOT/sa2va4b_caption_actor_bdd4"
GPU_ID="${CUDA_VISIBLE_DEVICES:-0}"
GPU_ID="${GPU_ID%%,*}"
REVISION=e870e79321c31b733e2031af5aa2fb1fe3ac7eec
SAMPLES=(
  val_bdd_pc_5k_video2556_phase2
  val_bdd_pc_5k_video8125_phase1
  val_bdd_pc_5k_video8139_phase2
)

echo "=== ProPainter BDD3 history background-plate smoke ==="
echo "Official ProPainter pinned at $REVISION; no submission frames are modified."
/workspace/wan_ticft/evaluation/check_gpu_free.sh "$GPU_ID" 22000

if [[ ! -d "$REPO/.git" ]]; then
  git clone https://github.com/sczhou/ProPainter.git "$REPO"
fi
git -C "$REPO" fetch --quiet origin "$REVISION"
git -C "$REPO" checkout --quiet "$REVISION"

if [[ ! -x "$ENV/bin/python" ]]; then
  "$PYTHON" -m venv --system-site-packages "$ENV"
fi
if ! "$ENV/bin/python" -c 'import addict, av, skimage, matplotlib, yapf' 2>/dev/null; then
  "$ENV/bin/pip" install --no-cache-dir addict av scikit-image matplotlib yapf
fi

PREP_ARGS=()
SHEET_ARGS=()
for sample_id in "${SAMPLES[@]}"; do
  PREP_ARGS+=(--sample_id "$sample_id")
  SHEET_ARGS+=(--sample_id "$sample_id")
done
"$PYTHON" -m wan_ticft.evaluation.prepare_propainter_history_smoke \
  --manifest "$MANIFEST" \
  --mask_root "$MASK_ROOT" \
  --out_dir "$OUT" \
  "${PREP_ARGS[@]}" \
  --pre_dilate 5 \
  --clean

mkdir -p "$OUT/inpaint"
for sample_id in "${SAMPLES[@]}"; do
  echo "[ProPainter] $sample_id"
  CUDA_VISIBLE_DEVICES="$GPU_ID" \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  "$ENV/bin/python" "$REPO/inference_propainter.py" \
    --video "$OUT/inputs/$sample_id" \
    --mask "$OUT/masks/$sample_id" \
    --output "$OUT/inpaint" \
    --neighbor_length 4 \
    --ref_stride 2 \
    --subvideo_length 20 \
    --mask_dilation 4 \
    --save_frames \
    --fp16
done

"$PYTHON" -m wan_ticft.evaluation.make_propainter_history_sheets \
  --root "$OUT" \
  "${SHEET_ARGS[@]}"

echo
echo "sheets=$OUT/sheets"
echo "Gate: the pedestrian must be fully removed in at least 2/3 samples, with plausible road/building"
echo "texture and no temporal flashing. Only then replace the depth renderer anchor in a scored gate."

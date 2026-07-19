#!/usr/bin/env bash
set -euo pipefail

cd /workspace

REPO=/workspace/third_party/vdpm
ENV=/workspace/experiments/wan_ticft/vdpm_env
OUT=/workspace/experiments/wan_ticft/vdpm_bdd_smoke4_motion
CHECKPOINT=/workspace/models/vdpm/model.pt
TOOLS=/workspace/experiments/wan_ticft/tools
UV=$TOOLS/bin/uv
GPU_ID="${CUDA_VISIBLE_DEVICES:-0}"
GPU_ID="${GPU_ID%%,*}"
MAX_FRAMES="${VDPM_MAX_FRAMES:-4}"

echo "=== V-DPM BDD dynamic-mask gate ==="
/workspace/wan_ticft/evaluation/check_gpu_free.sh "$GPU_ID" 22000
df -h /workspace

echo "=== Restore four-sample BDD inputs if cleanup removed them ==="
INPUT_COUNT=0
if [[ -d /workspace/experiments/wan_ticft/vdpm_bdd_smoke4/history_videos ]]; then
  INPUT_COUNT="$(find /workspace/experiments/wan_ticft/vdpm_bdd_smoke4/history_videos -maxdepth 1 -type f -name '*.mp4' | wc -l)"
fi
if (( INPUT_COUNT < 4 )); then
  /opt/conda/envs/wan22/bin/python -m wan_ticft.evaluation.prepare_vdpm_bdd_smoke \
    --ceiling_csv /workspace/experiments/wan_ticft/ltx_region_composite_ceiling.csv \
    --history_dir /workspace/experiments/wan_ticft/ltxv_bdd25_gate/history_videos \
    --out_dir /workspace/experiments/wan_ticft/vdpm_bdd_smoke4
fi

echo "=== Restore official V-DPM source if cleanup removed it ==="
if [[ ! -d "$REPO/.git" ]]; then
  rm -rf "$REPO"
  git clone --depth 1 https://github.com/eldar/vdpm.git "$REPO"
fi

if [[ ! -x "$UV" ]]; then
  mkdir -p "$TOOLS/bin"
  curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR="$TOOLS/bin" sh
fi

if [[ ! -x "$ENV/bin/python" ]]; then
  "$UV" venv --python 3.12 "$ENV"
fi

echo "=== Install isolated V-DPM environment ==="
"$UV" pip install --python "$ENV/bin/python" -r "$REPO/requirements.txt"

echo "=== Download official checkpoint ==="
mkdir -p "$(dirname "$CHECKPOINT")"
if [[ ! -s "$CHECKPOINT" ]]; then
  PARTIAL="$CHECKPOINT.partial"
  TORCH_PARTIAL="$(find "$(dirname "$CHECKPOINT")" -maxdepth 1 -type f \
    -name 'model.pt.*.partial' -print -quit)"
  if [[ ! -s "$PARTIAL" && -n "$TORCH_PARTIAL" ]]; then
    mv "$TORCH_PARTIAL" "$PARTIAL"
  fi
  curl \
    --location \
    --fail \
    --retry 30 \
    --retry-delay 5 \
    --retry-all-errors \
    --connect-timeout 30 \
    --speed-limit 1024 \
    --speed-time 120 \
    --continue-at - \
    --output "$PARTIAL" \
    "https://huggingface.co/edgarsucar/vdpm/resolve/main/model.pt"
  mv "$PARTIAL" "$CHECKPOINT"
fi
ls -lh "$CHECKPOINT"

echo "=== Run four-sample headless diagnostic ==="
TOKENIZERS_PARALLELISM=false "$ENV/bin/python" -m wan_ticft.evaluation.run_vdpm_motion_diagnostic \
  --input_dir /workspace/experiments/wan_ticft/vdpm_bdd_smoke4/history_videos \
  --out_dir "$OUT" \
  --vdpm_repo "$REPO" \
  --checkpoint "$CHECKPOINT" \
  --max_frames "$MAX_FRAMES" \
  --precision bf16

echo
echo "=== Gate result ==="
cat "$OUT/report.csv"
echo "Inspect motion overlays under: $OUT"
echo "Proceed only if motion is concentrated on true actors/ego-motion boundaries rather than the full scene."

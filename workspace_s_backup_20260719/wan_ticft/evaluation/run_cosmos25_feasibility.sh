#!/usr/bin/env bash
set -euo pipefail

cd /workspace

REPO=/workspace/third_party/cosmos-predict2.5
OUT=/workspace/experiments/wan_ticft/cosmos25_bdd_smoke4

echo "=== Cosmos-Predict2.5 feasibility audit ==="
echo "Kernel: $(uname -r)"
echo "glibc: $(ldd --version | head -1)"
echo "Driver:"
nvidia-smi --query-gpu=index,name,driver_version,memory.total,memory.free --format=csv,noheader
echo "Disk:"
df -h /workspace
echo "Docker: $(command -v docker || echo missing)"
echo "Git LFS: $(command -v git-lfs || echo missing)"
if [[ -n "${HF_TOKEN:-}" ]]; then
  echo "HF_TOKEN: set"
else
  echo "HF_TOKEN: not set"
fi

echo "=== Prepare four BDD history clips ==="
/opt/conda/envs/wan22/bin/python -m wan_ticft.evaluation.prepare_ati_bdd_smoke \
  --index_dir /workspace/experiments/index \
  --out_dir "$OUT" \
  --limit 4 \
  --fps 16

echo "=== Clone official Cosmos-Predict2.5 source only ==="
if [[ ! -d "$REPO/.git" ]]; then
  git clone --depth 1 https://github.com/nvidia-cosmos/cosmos-predict2.5.git "$REPO"
fi

echo
echo "=== Feasibility gate result ==="
echo "History clips: $OUT/history_videos"
echo "Repository: $REPO"
echo "No environment was installed and no checkpoint was downloaded."
echo "Next step only after reviewing setup size, HF license/token, and single-GPU memory path."

#!/usr/bin/env bash
set -euo pipefail

cd /workspace

PYTHON="${PYTHON:-/opt/conda/envs/wan22/bin/python}"
REPO="${DA3_REPO:-/workspace/third_party/depth-anything-3}"

echo "=== Setup Depth Anything 3 ==="
echo "repo=$REPO"
echo "python=$PYTHON"

mkdir -p /workspace/third_party
if [[ ! -d "$REPO/.git" ]]; then
  git clone https://github.com/ByteDance-Seed/depth-anything-3 "$REPO"
else
  echo "found existing repo; skip clone"
fi

"$PYTHON" -m pip install -U pip setuptools wheel
"$PYTHON" -m pip install -e "$REPO"

echo
echo "Depth Anything 3 setup complete."
echo "Try the gate:"
echo "  CUDA_VISIBLE_DEVICES=2 bash /workspace/wan_ticft/evaluation/run_da3_depth_layered_bdd24_gate.sh"

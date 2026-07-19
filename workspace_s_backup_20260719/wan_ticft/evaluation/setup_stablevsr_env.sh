#!/usr/bin/env bash
set -euo pipefail

cd /workspace

REPO="${STABLEVSR_REPO:-/workspace/third_party/StableVSR}"
ENV_PREFIX="${ENV_PREFIX:-/workspace/experiments/wan_ticft/stablevsr_env}"
PY="$ENV_PREFIX/bin/python"

echo "=== StableVSR setup ==="
echo "repo=$REPO"
echo "env=$ENV_PREFIX"

if [[ ! -d "$REPO/.git" ]]; then
  mkdir -p "$(dirname "$REPO")"
  git clone https://github.com/claudiom4sir/StableVSR.git "$REPO"
fi

if [[ ! -x "$PY" ]]; then
  conda create -y -p "$ENV_PREFIX" python=3.8.17
  "$PY" -m pip install --upgrade "pip<25" wheel setuptools
  conda install -y -p "$ENV_PREFIX" pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.7 -c pytorch -c nvidia
  "$PY" -m pip install -r "$REPO/requirements.txt"
  "$PY" -m pip install --force-reinstall "numpy==1.24.4" "setuptools<81"
fi

"$PY" - <<'PY'
import diffusers, torch, torchvision, numpy
print("torch", torch.__version__)
print("torchvision", torchvision.__version__)
print("diffusers", diffusers.__version__)
print("numpy", numpy.__version__)
PY

echo "StableVSR setup complete."

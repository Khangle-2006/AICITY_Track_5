#!/usr/bin/env bash
set -euo pipefail

cd /workspace

VISTA_REPO="${VISTA_REPO:-/workspace/third_party/Vista}"
ENV_PREFIX="${ENV_PREFIX:-/workspace/experiments/wan_ticft/vista_env}"
PY="$ENV_PREFIX/bin/python"

echo "=== Vista setup ==="
echo "repo=$VISTA_REPO"
echo "env=$ENV_PREFIX"

if [[ ! -d "$VISTA_REPO/.git" ]]; then
  mkdir -p "$(dirname "$VISTA_REPO")"
  git clone https://github.com/OpenDriveLab/Vista.git "$VISTA_REPO"
fi

if [[ ! -x "$PY" ]]; then
  conda create -y -p "$ENV_PREFIX" python=3.9
  "$PY" -m pip install --upgrade pip wheel setuptools
  conda install -y -p "$ENV_PREFIX" pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.7 -c pytorch -c nvidia
  "$PY" -m pip install -r "$VISTA_REPO/requirements.txt"
  "$PY" -m pip install --force-reinstall "numpy<2"
  "$PY" -m pip install "setuptools<81"
fi

mkdir -p "$VISTA_REPO/ckpts"
if [[ ! -s "$VISTA_REPO/ckpts/vista.safetensors" ]]; then
  "$PY" -m pip install -q huggingface_hub
  "$PY" - <<PY
from huggingface_hub import hf_hub_download
hf_hub_download(
    repo_id="OpenDriveLab/Vista",
    filename="vista.safetensors",
    local_dir="$VISTA_REPO/ckpts",
    local_dir_use_symlinks=False,
)
PY
fi

echo "Vista setup complete."
echo "checkpoint=$VISTA_REPO/ckpts/vista.safetensors"

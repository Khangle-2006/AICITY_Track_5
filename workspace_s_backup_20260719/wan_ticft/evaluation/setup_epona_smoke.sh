#!/usr/bin/env bash
set -euo pipefail

ROOT=/workspace/experiments/wan_ticft
ENV="$ROOT/epona_env"
PRETRAINED="$ROOT/epona_pretrained"
mkdir -p "$PRETRAINED"

if [[ ! -x "$ENV/bin/python" ]]; then
  /opt/conda/envs/wan22/bin/python -m venv --system-site-packages "$ENV"
fi

if [[ ! -f "$ENV/.epona_requirements_ready" ]] && \
   ! "$ENV/bin/python" -c 'import transformers, einops, cv2, omegaconf, mmengine, safetensors' >/dev/null 2>&1; then
  "$ENV/bin/python" -m pip install --upgrade pip
  "$ENV/bin/python" -m pip install \
    transformers==4.39.3 einops==0.6.1 numpy==1.26.4 scipy==1.13.0 \
    opencv-python==4.10.0.84 Pillow==10.3.0 imageio==2.34.1 \
    matplotlib==3.5.3 tqdm==4.66.4 omegaconf==2.3.0 addict==2.4.0 \
    mmengine==0.10.3 safetensors==0.4.2 PyYAML==6.0.1
fi
touch "$ENV/.epona_requirements_ready"

download() {
  local name="$1"
  local path="$PRETRAINED/$name"
  local size=0
  [[ -f "$path" ]] && size="$(stat -c '%s' "$path")"
  if (( size < 100000000 )); then
    rm -f "$PRETRAINED/$name"
    FILENAME="$name" PRETRAINED="$PRETRAINED" "$ENV/bin/python" - <<'PY'
import os
from huggingface_hub import hf_hub_download

path = hf_hub_download(
    repo_id="Kevin-thu/Epona",
    filename=os.environ["FILENAME"],
    local_dir=os.environ["PRETRAINED"],
)
print(f"downloaded {path}")
PY
  fi
}

download epona_nuplan+nusc.pkl
download dcae_td_20000.pkl

echo "Epona smoke environment and checkpoints are ready."

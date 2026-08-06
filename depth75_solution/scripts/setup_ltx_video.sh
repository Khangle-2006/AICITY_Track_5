#!/usr/bin/env bash
set -euo pipefail

SOLUTION_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
THIRD_PARTY=${THIRD_PARTY:-$SOLUTION_DIR/third_party}
LTX_REPO=${LTX_REPO:-$THIRD_PARTY/LTX-Video}
LTX_ENV=${LTX_ENV:-$THIRD_PARTY/ltxv_env}
PYTHON=${PYTHON:-python3.10}
MODEL=ltxv-2b-0.9.6-distilled-04-25.safetensors

mkdir -p "$THIRD_PARTY"
if [[ ! -d "$LTX_REPO/.git" ]]; then
  git clone https://github.com/Lightricks/LTX-Video.git "$LTX_REPO"
fi

if [[ ! -x "$LTX_ENV/bin/python" ]]; then
  "$PYTHON" -m venv "$LTX_ENV"
fi

"$LTX_ENV/bin/python" -m pip install --upgrade pip
"$LTX_ENV/bin/python" -m pip install -e "$LTX_REPO[inference]"

if [[ ! -f "$LTX_REPO/$MODEL" ]]; then
  "$LTX_ENV/bin/hf" download Lightricks/LTX-Video "$MODEL" --local-dir "$LTX_REPO"
fi

echo
echo "LTX_REPO=$LTX_REPO"
echo "LTX_PYTHON=$LTX_ENV/bin/python"
echo "PIPELINE_CONFIG=$SOLUTION_DIR/wan_ticft/evaluation/ltxv_2b_gate.yaml"

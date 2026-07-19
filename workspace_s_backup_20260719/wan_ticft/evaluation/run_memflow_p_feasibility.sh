#!/usr/bin/env bash
set -euo pipefail

cd /workspace

REPO=/workspace/third_party/MemFlow
OUT=/workspace/experiments/wan_ticft/memflow_p_feasibility.txt

{
  echo "=== MemFlow-P future-flow feasibility audit ==="
  echo "date=$(date -Iseconds)"
  echo

  if [[ ! -d "$REPO/.git" ]]; then
    echo "status=cloning_official_source_only"
    echo "repo=$REPO"
    git clone --depth 1 https://github.com/DQiaole/MemFlow.git "$REPO"
  fi

  echo "status=source_present"
  git -C "$REPO" rev-parse HEAD
  echo
  echo "prediction_files:"
  find "$REPO" -maxdepth 2 -type f \
    \( -name '*predict*' -o -name '*MemFlowNet_P*' -o -name 'README.md' \) \
    -printf '%p\n' | sort
  echo
  echo "prediction_references:"
  grep -RIn --exclude-dir=.git \
    -E 'MemFlowNet_P|future prediction|future_prediction|predict' "$REPO" \
    | head -120 || true
  echo
  echo "checkpoint_files:"
  find "$REPO" -maxdepth 3 -type f \
    \( -name '*.pth' -o -name '*.pt' -o -name '*.ckpt' \) \
    -printf '%p | %s bytes\n' | sort
  echo
  echo "environment:"
  /opt/conda/envs/wan22/bin/python - <<'PY'
import sys
print("python", sys.version.replace("\n", " "))
try:
    import torch
    print("torch", torch.__version__)
    print("cuda", torch.version.cuda)
except Exception as exc:
    print("torch_error", repr(exc))
PY
} | tee "$OUT"

echo
echo "Gate rule:"
echo "  - Pass only if official MemFlow-P inference is directly callable on history frames"
echo "    and requires no training or invasive environment changes."
echo "  - After feasibility pass, allow one checkpoint download and one 2-sample BDD"
echo "    one-step/multi-step rollout smoke."
echo "  - Stop immediately if only one-step prediction is usable, rollout is unsupported,"
echo "    or the environment requires substantial repair."
echo "output $OUT"

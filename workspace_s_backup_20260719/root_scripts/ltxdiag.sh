#!/usr/bin/env bash
set -euo pipefail

PYTHONPATH=/workspace/third_party/LTX-Video \
/workspace/experiments/wan_ticft/ltxv_env/bin/python \
  -m wan_ticft.evaluation.diagnose_ltxv_val_input

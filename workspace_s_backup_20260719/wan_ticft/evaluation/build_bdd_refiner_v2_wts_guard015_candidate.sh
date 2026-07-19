#!/usr/bin/env bash
set -euo pipefail

echo "=== Build BDD refiner-router v2 + WTS guard015 candidate ==="
echo "This is a model-level/classical+learned BDD candidate, not CNN/LTX/Cosmos blend."

cd /workspace
PYTHON=${PYTHON:-/opt/conda/envs/wan22/bin/python}
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export CUDA_VISIBLE_DEVICES
export PYTHONPATH=/workspace/wan_ticft:${PYTHONPATH:-}

INDEX=${INDEX:-/workspace/experiments/index}
CHECKPOINT=${CHECKPOINT:-/workspace/experiments/refiner_bdd_direct_h1_120_v2/refiner_best.pt}
BDD_OUT=${BDD_OUT:-/workspace/experiments/wan_ticft/bdd_refiner_router_v2_test}
WTS_DIR=${WTS_DIR:-/workspace/experiments/wan_ticft/wts_guard015_test}
SUBMISSION=${SUBMISSION:-/workspace/experiments/hybrid_bdd_refiner_v2_wts_guard015_candidate}
REPORT=${REPORT:-/workspace/experiments/wan_ticft/hybrid_bdd_refiner_v2_wts_guard015_candidate_report.csv}

echo "checkpoint=$CHECKPOINT"
echo "bdd_out=$BDD_OUT"
echo "wts=$WTS_DIR"
echo "submission=$SUBMISSION"

if [[ ! -f "$CHECKPOINT" ]]; then
  echo "Missing checkpoint: $CHECKPOINT" >&2
  exit 2
fi
if [[ ! -d "$WTS_DIR" ]]; then
  echo "Missing WTS dir: $WTS_DIR" >&2
  exit 2
fi

"$PYTHON" -m wan_ticft.inference.run_bdd_refiner_router_v2 \
  --index_dir "$INDEX" \
  --checkpoint "$CHECKPOINT" \
  --out "$BDD_OUT" \
  --clean

"$PYTHON" -m wan_ticft.evaluation.build_hybrid_submission \
  --index_dir "$INDEX" \
  --wan_dir "$BDD_OUT" \
  --region_dir "$WTS_DIR" \
  --out_dir "$SUBMISSION" \
  --report_csv "$REPORT" \
  --wan_label bdd_refiner_router_v2 \
  --region_label wts_guard015 \
  --region_views wts_overhead_or_fixed_camera wts_vehicle_or_ip_camera \
  --clean

"$PYTHON" -m baseline.submission \
  --index_dir "$INDEX" \
  --gen_dir "$SUBMISSION" \
  --submission_dir "$SUBMISSION" \
  --verify_only

cat <<EOF

Candidate verified:
  $SUBMISSION
Report:
  $REPORT

Robocopy:
  robocopy S:\\experiments\\$(basename "$SUBMISSION") G:\\$(basename "$SUBMISSION") /E /MT:16 /R:2 /W:2
EOF

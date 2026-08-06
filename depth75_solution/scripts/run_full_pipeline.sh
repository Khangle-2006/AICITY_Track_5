#!/usr/bin/env bash
set -euo pipefail

SOLUTION_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="$SOLUTION_DIR:${PYTHONPATH:-}"

PYTHON=${PYTHON:-python}
MODE=${MODE:-exact}
WORK_ROOT=${WORK_ROOT:-$SOLUTION_DIR/outputs}
INDEX_DIR=${INDEX_DIR:-$WORK_ROOT/index}
TEST_CSV=${TEST_CSV:-$INDEX_DIR/test_index.csv}
DATA_ROOT=${DATA_ROOT:-}
ROOT=${ROOT:-$WORK_ROOT/wan_ticft}
LTXHF_BASE=${LTXHF_BASE:-$ROOT/ltxhf_t55_bdd_test}
SUBMISSION=${SUBMISSION:-$WORK_ROOT/hybrid_multiframe_mf_h8_ltxhf_t55_wts_candidate}
ZIP_PATH=${ZIP_PATH:-${SUBMISSION}.zip}
SEED=${SEED:-17}

usage() {
  cat <<'EOF'
Usage:
  MODE=exact LTXHF_BASE=/path/to/archived/ltxhf_t55_bdd_test \
    DATA_ROOT=/path/to/Track_5 bash scripts/run_full_pipeline.sh

  MODE=regenerate LTX_REPO=/path/to/LTX-Video LTX_PYTHON=/path/to/ltx/python \
    PIPELINE_CONFIG=/path/to/ltxv-2b-0.9.6-distilled.yaml \
    DATA_ROOT=/path/to/Track_5 bash scripts/run_full_pipeline.sh

MODE=exact reuses the archived BDD base that produced the leaderboard entry.
MODE=regenerate creates a new LTX base and traffic-detail donor before running
the deterministic T55, mf_h8, WTS, assembly, verification, and zip stages.
EOF
}

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  usage
  exit 0
fi
if [[ "$MODE" != "exact" && "$MODE" != "regenerate" ]]; then
  echo "MODE must be exact or regenerate" >&2
  exit 2
fi

mkdir -p "$INDEX_DIR" "$ROOT"
if [[ ! -f "$TEST_CSV" ]]; then
  if [[ -z "$DATA_ROOT" ]]; then
    echo "Set DATA_ROOT or provide TEST_CSV=$TEST_CSV" >&2
    exit 2
  fi
  "$PYTHON" "$SOLUTION_DIR/scripts/relocate_test_index.py" \
    --template "$SOLUTION_DIR/test_index_hybrid.csv" \
    --data-root "$DATA_ROOT" \
    --output "$TEST_CSV"
fi

if [[ "$MODE" == "regenerate" ]]; then
  : "${LTX_REPO:?Set LTX_REPO to the official LTX-Video checkout}"
  : "${LTX_PYTHON:?Set LTX_PYTHON to the LTX-Video environment Python}"
  PIPELINE_CONFIG=${PIPELINE_CONFIG:-$SOLUTION_DIR/wan_ticft/evaluation/ltxv_2b_gate.yaml}
  BASE_BDD=${BASE_BDD:-$ROOT/ltx_base_bdd_test}
  TRAFFIC_DONOR=${TRAFFIC_DONOR:-$ROOT/ltx_traffic_detail_bdd_test}

  echo "[1/6] Generate the default LTX BDD future sequence"
  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} "$PYTHON" -m wan_ticft.evaluation.run_ltxv_test_bdd \
    --test_csv "$TEST_CSV" \
    --out_dir "$BASE_BDD" \
    --ltx_repo "$LTX_REPO" \
    --ltx_python "$LTX_PYTHON" \
    --pipeline_config "$PIPELINE_CONFIG" \
    --seed "$SEED" \
    --prompt_mode legacy \
    --resume

  echo "[2/6] Generate the traffic-detail LTX donor"
  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} "$PYTHON" -m wan_ticft.evaluation.run_ltxv_test_bdd \
    --test_csv "$TEST_CSV" \
    --out_dir "$TRAFFIC_DONOR" \
    --ltx_repo "$LTX_REPO" \
    --ltx_python "$LTX_PYTHON" \
    --pipeline_config "$PIPELINE_CONFIG" \
    --seed "$SEED" \
    --prompt_mode traffic_detail \
    --resume

  echo "[3/6] Transfer traffic-detail high frequencies at alpha=0.55"
  "$PYTHON" -m wan_ticft.evaluation.ltx_detail_donor_test \
    --index_csv "$TEST_CSV" \
    --base_dir "$BASE_BDD" \
    --donor_dir "$TRAFFIC_DONOR" \
    --out_dir "$LTXHF_BASE" \
    --alpha 0.55 \
    --sigma 2.2 \
    --clean
else
  echo "[1-3/6] Reuse archived LTXHF-T55 base: $LTXHF_BASE"
  if [[ ! -f "$LTXHF_BASE/report.csv" ]]; then
    echo "Missing exact base report: $LTXHF_BASE/report.csv" >&2
    exit 2
  fi
fi

echo "[4/6] Run mf_h8 geometry refinement and WTS specialists"
INDEX="$INDEX_DIR" \
TEST_CSV="$TEST_CSV" \
ROOT="$ROOT" \
LTXHF_BASE="$LTXHF_BASE" \
SUBMISSION="$SUBMISSION" \
PYTHON="$PYTHON" \
bash "$SOLUTION_DIR/scripts/build_multiframe_ltxhf_t55_candidate.sh"

echo "[5/6] Validate the complete 71-video candidate"
"$PYTHON" "$SOLUTION_DIR/scripts/validate_candidate_report.py" \
  "$ROOT/hybrid_multiframe_mf_h8_ltxhf_t55_wts_candidate_report.csv" \
  --expected-samples 71 \
  --expected-frames 5194

echo "[6/6] Create the submission archive"
INDEX="$INDEX_DIR" \
SUBMISSION="$SUBMISSION" \
ZIP_PATH="$ZIP_PATH" \
PYTHON="$PYTHON" \
bash "$SOLUTION_DIR/scripts/create_submission_zip.sh"

echo
echo "submission=$SUBMISSION"
echo "zip=$ZIP_PATH"

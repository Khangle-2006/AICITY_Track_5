#!/usr/bin/env bash
set -euo pipefail

cd /workspace

PYTHON=${PYTHON:-/opt/conda/envs/wan22/bin/python}
INDEX=${INDEX:-/workspace/experiments/index}
TEST_CSV=${TEST_CSV:-$INDEX/test_index.csv}
ROOT=${ROOT:-/workspace/experiments/wan_ticft}
LTX_RAW=${LTX_RAW:-$ROOT/ltxv_bdd_test_full}
BDD16=${BDD16:-$ROOT/ltxv_bdd_test_transition16}
BDD32=${BDD32:-$ROOT/ltxv_bdd_test_transition32}
BDD_ROUTE=${BDD_ROUTE:-$ROOT/ltxv_bdd_test_phase0_transition32_route}
WTS=${WTS:-$ROOT/wts_overhead_long24_vehicle_medium_test_phase0_route32}
SUBMISSION=${SUBMISSION:-/workspace/experiments/hybrid_ltx_phase0_transition32_wts_domain_routed_candidate}
REPORT=${REPORT:-/workspace/experiments/wan_ticft/hybrid_ltx_phase0_transition32_wts_domain_routed_candidate_report.csv}

echo "=== Build BDD phase0 transition32 + WTS current routed candidate ==="
echo "LTX_RAW=${LTX_RAW}"
echo "BDD16=${BDD16}"
echo "BDD32=${BDD32}"
echo "BDD_ROUTE=${BDD_ROUTE}"
echo "WTS=${WTS}"
echo "SUBMISSION=${SUBMISSION}"

if [[ ! -d "${LTX_RAW}" ]]; then
  echo "Missing LTX_RAW: ${LTX_RAW}" >&2
  exit 2
fi
if [[ ! -d "${BDD16}" ]]; then
  echo "Missing BDD16: ${BDD16}" >&2
  exit 2
fi

if [[ ! -d "${BDD32}" ]]; then
  echo
  echo "[1/6] Build BDD transition32 from raw LTX test output"
  "${PYTHON}" -m wan_ticft.evaluation.build_ltx_test_transition \
    --test_csv "${TEST_CSV}" \
    --ltx_dir "${LTX_RAW}" \
    --out_dir "${BDD32}" \
    --transition_frames 32 \
    --curve linear \
    --clean
else
  echo
  echo "[1/6] Reuse existing BDD transition32: ${BDD32}"
fi

echo
echo "[2/6] Build BDD phase route: phase0 transition32, phase1-4 transition16"
rm -rf "${BDD_ROUTE}"
export TEST_CSV BDD16 BDD32 BDD_ROUTE
"${PYTHON}" - <<'PY'
import csv
import os
import shutil
from pathlib import Path

from baseline.frame_io import numeric_pngs
from baseline.verify import verify_sample_output

test_csv = Path(os.environ["TEST_CSV"])
bdd16 = Path(os.environ["BDD16"])
bdd32 = Path(os.environ["BDD32"])
out = Path(os.environ["BDD_ROUTE"])

out.mkdir(parents=True, exist_ok=True)
rows = []
with test_csv.open(encoding="utf-8", newline="") as handle:
    for row in csv.DictReader(handle):
        if row.get("view_type") == "bdd":
            rows.append(row)

report = []
for position, row in enumerate(rows, start=1):
    output_name = row["source_folder"]
    target_n = int(row["target_N"])
    source_root = bdd32 if str(row.get("phase", "")) == "0" else bdd16
    source = source_root / output_name
    destination = out / output_name
    verified = verify_sample_output(source, target_n)
    if not verified.ok:
        raise RuntimeError(f"{output_name}: bad source {source}: {verified.notes}")
    destination.mkdir(parents=True, exist_ok=True)
    for png in numeric_pngs(source):
        target = destination / f"{int(png.stem)}.png"
        try:
            os.link(png, target)
        except OSError:
            shutil.copy2(png, target)
    verified_out = verify_sample_output(destination, target_n)
    if not verified_out.ok:
        raise RuntimeError(f"{output_name}: bad route output: {verified_out.notes}")
    route = "transition32" if source_root == bdd32 else "transition16"
    report.append(
        {
            "output_name": output_name,
            "phase": row.get("phase", ""),
            "route": route,
            "target_N": target_n,
            "frames_found": verified_out.frames_found,
            "status": "ok",
        }
    )
    print(f"[{position}/{len(rows)}] {output_name} | {route} | ok", flush=True)

with (out / "report.csv").open("w", encoding="utf-8", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=report[0].keys())
    writer.writeheader()
    writer.writerows(report)
print(f"wrote {out}")
PY

echo
echo "[3/6] Generate WTS overhead/fixed current route: long24"
rm -rf "${WTS}" "${SUBMISSION}"
"${PYTHON}" -m baseline.cli \
  --index_dir "${INDEX}" \
  --out "${WTS}" \
  --view_type wts_overhead_or_fixed_camera \
  --method region_norm_medium \
  --region_norm_decay 0.995 \
  --region_norm_max_flow_scale 24.0

echo
echo "[4/6] Generate WTS vehicle/IP current route: medium"
"${PYTHON}" -m baseline.cli \
  --index_dir "${INDEX}" \
  --out "${WTS}" \
  --view_type wts_vehicle_or_ip_camera \
  --method region_norm_medium \
  --region_norm_decay 0.96 \
  --region_norm_max_flow_scale 6.0

echo
echo "[5/6] Build hybrid submission directory, no zip"
"${PYTHON}" -m wan_ticft.evaluation.build_hybrid_submission \
  --index_dir "${INDEX}" \
  --wan_dir "${BDD_ROUTE}" \
  --region_dir "${WTS}" \
  --out_dir "${SUBMISSION}" \
  --report_csv "${REPORT}" \
  --wan_label ltx_phase0_transition32 \
  --region_label wts_overhead_long24_vehicle_medium \
  --region_views wts_overhead_or_fixed_camera wts_vehicle_or_ip_camera \
  --clean

echo
echo "[6/6] Verify full submission folder"
"${PYTHON}" -m baseline.submission \
  --index_dir "${INDEX}" \
  --gen_dir "${SUBMISSION}" \
  --submission_dir "${SUBMISSION}" \
  --verify_only

echo
echo "Candidate verified but not zipped:"
echo "  ${SUBMISSION}"
echo "Report:"
echo "  ${REPORT}"

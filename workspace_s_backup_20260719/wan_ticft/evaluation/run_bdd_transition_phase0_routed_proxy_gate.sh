#!/usr/bin/env bash
set -euo pipefail

cd /workspace

PYTHON=${PYTHON:-/opt/conda/envs/wan22/bin/python}
ROOT=${ROOT:-/workspace/experiments/wan_ticft}
MANIFEST=${MANIFEST:-$ROOT/ltxv_bdd25_gate/dense40_manifest.jsonl}
TRANSITION16=${TRANSITION16:-$ROOT/ltxv_bdd25_transitions/ltx_transition_16/report.csv}
TRANSITION24=${TRANSITION24:-$ROOT/ltxv_bdd25_transition_duration_gate/ltx_transition_24/report.csv}
TRANSITION32=${TRANSITION32:-$ROOT/ltxv_bdd25_transition_duration_gate/ltx_transition_32/report.csv}
ROUTE24=${ROUTE24:-$ROOT/bdd_transition_phase0_route24_report.csv}
ROUTE32=${ROUTE32:-$ROOT/bdd_transition_phase0_route32_report.csv}
OUT_CSV=${OUT_CSV:-$ROOT/bdd_transition_phase0_routed_perceptual_proxy.csv}
GPU_ID="${CUDA_VISIBLE_DEVICES:-3}"
GPU_ID="${GPU_ID%%,*}"

echo "=== BDD transition phase0 routed aggregate proxy ==="
echo "Route24: phase0 -> transition24, phase1-4 -> transition16"
echo "Route32: phase0 -> transition32, phase1-4 -> transition16"

export MANIFEST TRANSITION16 TRANSITION24 TRANSITION32 ROUTE24 ROUTE32

"${PYTHON}" - <<'PY'
import csv
import json
import os
from pathlib import Path

manifest = Path(os.environ["MANIFEST"])
transition16 = Path(os.environ["TRANSITION16"])
transition24 = Path(os.environ["TRANSITION24"])
transition32 = Path(os.environ["TRANSITION32"])
route24 = Path(os.environ["ROUTE24"])
route32 = Path(os.environ["ROUTE32"])


def read_manifest(path):
    out = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            out[row["sample_id"]] = str(row.get("phase", ""))
    return out


def read_report(path):
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return {row["sample_id"]: row for row in rows}


def write_route(path, base, phase0):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(base[next(iter(base))].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for sample_id in sorted(base):
            row = dict(phase0[sample_id] if phases.get(sample_id) == "0" else base[sample_id])
            writer.writerow({field: row.get(field, "") for field in fields})


phases = read_manifest(manifest)
base = read_report(transition16)
t24 = read_report(transition24)
t32 = read_report(transition32)
missing24 = sorted(sample for sample in base if phases.get(sample) == "0" and sample not in t24)
missing32 = sorted(sample for sample in base if phases.get(sample) == "0" and sample not in t32)
if missing24 or missing32:
    raise SystemExit(f"missing phase0 rows: t24={missing24[:3]} t32={missing32[:3]}")

write_route(route24, base, t24)
write_route(route32, base, t32)
print(f"wrote {route24}")
print(f"wrote {route32}")
PY

/workspace/wan_ticft/evaluation/check_gpu_free.sh "$GPU_ID" 3000
CUDA_VISIBLE_DEVICES="$GPU_ID" \
TORCH_HOME="$ROOT/tools/torch_cache" \
"${PYTHON}" -m wan_ticft.evaluation.audit_perceptual_video_distribution \
  --manifest "$MANIFEST" \
  --report "transition16=$TRANSITION16" \
  --report "phase0_route24=$ROUTE24" \
  --report "phase0_route32=$ROUTE32" \
  --out_csv "$OUT_CSV"

echo
echo "Gate rule:"
echo "  - Promote only if a routed candidate improves both aggregate FID and FVD"
echo "    proxies versus transition16."
echo "  - Prefer route24 over route32 if FVD is similar because phase0 had the"
echo "    better phase-local FVD and shorter blend is less visually risky."
echo "  - If neither route improves both proxies, keep transition16 globally."

"""Build a report by selecting rows from one report for selected phases."""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path


def read_rows(path: Path) -> dict[str, dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return {row["sample_id"]: row for row in csv.DictReader(handle) if row.get("status") == "ok"}


def phase_of(sample_id: str) -> str:
    match = re.search(r"phase(\d+)", sample_id)
    return match.group(1) if match else ""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base_report", type=Path, required=True)
    parser.add_argument("--phase_report", type=Path, required=True)
    parser.add_argument("--phase", action="append", required=True)
    parser.add_argument("--route_label", default="")
    parser.add_argument("--out_report", type=Path, required=True)
    args = parser.parse_args()

    base = read_rows(args.base_report)
    phase_rows = read_rows(args.phase_report)
    selected = set(str(item) for item in args.phase)
    rows = []
    for sample_id, base_row in base.items():
        use_phase = phase_of(sample_id) in selected and sample_id in phase_rows
        out = dict(phase_rows[sample_id] if use_phase else base_row)
        out["route"] = args.route_label or ("phase_" + "_".join(sorted(selected)))
        rows.append(out)

    fields = []
    for row in rows:
        for key in row.keys():
            if key not in fields:
                fields.append(key)
    args.out_report.parent.mkdir(parents=True, exist_ok=True)
    with args.out_report.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"rows={len(rows)}")
    print(f"output={args.out_report}")


if __name__ == "__main__":
    main()

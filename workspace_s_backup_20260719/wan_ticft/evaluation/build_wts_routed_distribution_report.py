"""Build a WTS validation video report routed by camera domain."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def read_rows(path: Path) -> dict[str, dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return {row["sample_id"]: row for row in csv.DictReader(handle)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--medium_report", type=Path, required=True)
    parser.add_argument("--long24_report", type=Path, required=True)
    parser.add_argument("--out_report", type=Path, required=True)
    args = parser.parse_args()

    medium = read_rows(args.medium_report)
    long24 = read_rows(args.long24_report)
    output = []
    for sample_id, medium_row in medium.items():
        if medium_row["view_type"] == "wts_overhead_or_fixed_camera":
            row = dict(long24[sample_id])
            row["route"] = "region_long24"
        else:
            row = dict(medium_row)
            row["route"] = "region_medium"
        output.append(row)

    args.out_report.parent.mkdir(parents=True, exist_ok=True)
    fields = [*next(iter(output)).keys()]
    if "route" not in fields:
        fields.append("route")
    with args.out_report.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(output)
    print(f"rows={len(output)}")
    print(f"output={args.out_report}")


if __name__ == "__main__":
    main()

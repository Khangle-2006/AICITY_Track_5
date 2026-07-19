"""Build a WTS report by selecting one report per WTS view type."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def read_rows(path: Path) -> dict[str, dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return {row["sample_id"]: row for row in csv.DictReader(handle) if row.get("status") == "ok"}


def row_view_type(row: dict[str, str]) -> str:
    view_type = row.get("view_type", "")
    if view_type:
        return view_type
    sample_id = row.get("sample_id", "")
    if "wts_overhead_or_fixed_camera" in sample_id:
        return "wts_overhead_or_fixed_camera"
    if "wts_vehicle_or_ip_camera" in sample_id:
        return "wts_vehicle_or_ip_camera"
    return ""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--overhead_report", type=Path, required=True)
    parser.add_argument("--vehicle_report", type=Path, required=True)
    parser.add_argument("--out_report", type=Path, required=True)
    parser.add_argument("--route_label", default="")
    args = parser.parse_args()

    rows = []
    for report, expected_view in (
        (args.overhead_report, "wts_overhead_or_fixed_camera"),
        (args.vehicle_report, "wts_vehicle_or_ip_camera"),
    ):
        for row in read_rows(report).values():
            view_type = row_view_type(row)
            if view_type != expected_view:
                continue
            out = dict(row)
            out["view_type"] = view_type
            out["route"] = args.route_label or report.parent.name
            rows.append(out)

    if not rows:
        raise RuntimeError("no rows selected")
    args.out_report.parent.mkdir(parents=True, exist_ok=True)
    fields = []
    for row in rows:
        for key in row.keys():
            if key not in fields:
                fields.append(key)
    with args.out_report.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"rows={len(rows)}")
    print(f"output={args.out_report}")


if __name__ == "__main__":
    main()

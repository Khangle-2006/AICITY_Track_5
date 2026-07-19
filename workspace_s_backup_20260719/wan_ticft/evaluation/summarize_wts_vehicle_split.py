"""Summarize WTS vehicle/IP test split and optional candidate route report."""

from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path


def is_normal(source_folder: str) -> bool:
    return "normal" in source_folder


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test_csv", type=Path, default=Path("/workspace/experiments/index/test_index.csv"))
    parser.add_argument("--report_csv", type=Path, default=None)
    args = parser.parse_args()

    with args.test_csv.open(encoding="utf-8", newline="") as handle:
        test_rows = list(csv.DictReader(handle))

    wts_rows = [row for row in test_rows if row.get("view_type", "").startswith("wts")]
    vehicle_rows = [row for row in wts_rows if row.get("view_type") == "wts_vehicle_or_ip_camera"]

    print("WTS test rows")
    print("-" * 72)
    by_view = Counter(row.get("view_type", "") for row in wts_rows)
    for view, count in sorted(by_view.items()):
        print(f"{view:<36} {count:>4}")

    print()
    print("WTS vehicle/IP source split")
    print("-" * 72)
    by_split = Counter("normal" if is_normal(row.get("source_folder", "")) else "event_ip" for row in vehicle_rows)
    for split, count in sorted(by_split.items()):
        print(f"{split:<36} {count:>4}")

    for split in ("event_ip", "normal"):
        print()
        print(split)
        print("-" * 72)
        for row in vehicle_rows:
            source_folder = row.get("source_folder", "")
            if ("normal" if is_normal(source_folder) else "event_ip") != split:
                continue
            print(f"{row.get('phase',''):<6} {source_folder}")

    if not args.report_csv:
        return

    with args.report_csv.open(encoding="utf-8", newline="") as handle:
        report_rows = list(csv.DictReader(handle))
    report_vehicle = [
        row for row in report_rows if row.get("view_type") == "wts_vehicle_or_ip_camera"
    ]
    routed = defaultdict(Counter)
    for row in report_vehicle:
        split = "normal" if is_normal(row.get("source_folder", "")) else "event_ip"
        routed[split][row.get("source_choice", "")] += 1

    print()
    print("Route report source choices")
    print("-" * 72)
    for split in ("event_ip", "normal"):
        for source_choice, count in sorted(routed[split].items()):
            print(f"{split:<12} {source_choice:<44} {count:>4}")


if __name__ == "__main__":
    main()

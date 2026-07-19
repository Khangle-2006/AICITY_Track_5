"""Route WTS vehicle/IP validation reports by normal-vs-event source subtype."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def read_manifest(path: Path) -> dict[str, dict]:
    rows = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            rows[row["sample_id"]] = row
    return rows


def read_report(path: Path) -> dict[str, dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return {
            row["sample_id"]: row
            for row in csv.DictReader(handle)
            if row.get("status") == "ok"
        }


def is_normal(row: dict) -> bool:
    return "normal" in str(row.get("source_folder", ""))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--base_report", type=Path, required=True)
    parser.add_argument("--depth_report", type=Path, required=True)
    parser.add_argument("--route", choices=["current", "normal_only", "event_only", "all_depth"], required=True)
    parser.add_argument("--out_report", type=Path, required=True)
    args = parser.parse_args()

    manifest = read_manifest(args.manifest)
    base = read_report(args.base_report)
    depth = read_report(args.depth_report)
    rows = []
    for sample_id, meta in manifest.items():
        normal = is_normal(meta)
        if args.route == "current":
            source = base
            source_choice = "current_base"
        elif args.route == "all_depth":
            source = depth
            source_choice = "depth_all"
        elif args.route == "normal_only" and normal:
            source = depth
            source_choice = "depth_normal"
        elif args.route == "event_only" and not normal:
            source = depth
            source_choice = "depth_event_ip"
        else:
            source = base
            source_choice = "current_base"
        if sample_id not in source:
            raise RuntimeError(f"{sample_id}: missing from selected source report")
        row = dict(source[sample_id])
        row["dataset_part"] = meta.get("dataset_part", "")
        row["view_type"] = meta.get("view_type", "")
        row["phase"] = str(meta.get("phase", ""))
        row["source_folder"] = meta.get("source_folder", "")
        row["source_split"] = "normal" if normal else "event_ip"
        row["source_choice"] = source_choice
        row["route"] = args.route
        rows.append(row)

    if not rows:
        raise RuntimeError("no rows selected")
    args.out_report.parent.mkdir(parents=True, exist_ok=True)
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with args.out_report.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    counts = {}
    for row in rows:
        key = (row["source_split"], row["source_choice"])
        counts[key] = counts.get(key, 0) + 1
    print(f"rows={len(rows)}")
    for (split, source_choice), count in sorted(counts.items()):
        print(f"{split} {source_choice}: {count}")
    print(f"output={args.out_report}")


if __name__ == "__main__":
    main()

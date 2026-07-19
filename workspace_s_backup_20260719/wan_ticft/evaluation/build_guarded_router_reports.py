"""Build conservative variants from a supervised candidate-router report."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def read_manifest(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def read_csv_map(path: Path, key: str = "sample_id") -> dict[str, dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return {row[key]: row for row in csv.DictReader(handle) if row.get(key)}


def parse_report(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise ValueError(f"expected NAME=CSV, got {value!r}")
    name, raw_path = value.split("=", 1)
    return name, Path(raw_path)


def write_report(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["sample_id", "video", "status", "chosen", "variant"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({field: row.get(field, "") for field in fields} for row in rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--choices", type=Path, required=True)
    parser.add_argument("--report", action="append", required=True, help="NAME=CSV")
    parser.add_argument("--out_dir", type=Path, required=True)
    args = parser.parse_args()

    rows = read_manifest(args.manifest)
    choices = read_csv_map(args.choices)
    reports = {name: read_csv_map(path) for name, path in map(parse_report, args.report)}
    required = {"depth_a75", "repeat_last", "region_medium", "region_long24", "actor_small"}
    missing = sorted(required - set(reports))
    if missing:
        raise ValueError(f"missing reports: {missing}")

    variants: dict[str, list[dict[str, str]]] = {
        "router_medium_guard": [],
        "phase0_depth_router_medium_guard": [],
        "phase0_depth_phase12_repeat_router_medium_guard": [],
    }

    def choose_video(sample_id: str, candidate: str, variant: str) -> dict[str, str]:
        row = reports[candidate].get(sample_id)
        if row is None:
            row = reports["depth_a75"].get(sample_id) or reports["repeat_last"].get(sample_id)
            candidate = "depth_a75_fallback" if row else candidate
        return {
            "sample_id": sample_id,
            "video": row.get("video", "") if row else "",
            "status": "ok" if row and row.get("video") else "bad",
            "chosen": candidate,
            "variant": variant,
        }

    for row in rows:
        sample_id = row["sample_id"]
        phase = int(row.get("phase") or 0)
        chosen = choices.get(sample_id, {}).get("chosen") or "depth_a75"

        medium_guard = "region_medium" if chosen == "region_long24" else chosen
        variants["router_medium_guard"].append(
            choose_video(sample_id, medium_guard, "router_medium_guard")
        )

        phase0_guard = "depth_a75" if phase == 0 else medium_guard
        variants["phase0_depth_router_medium_guard"].append(
            choose_video(sample_id, phase0_guard, "phase0_depth_router_medium_guard")
        )

        phase12_repeat_guard = phase0_guard
        if phase in {1, 2} and medium_guard in {"region_medium", "actor_small"}:
            phase12_repeat_guard = "repeat_last"
        variants["phase0_depth_phase12_repeat_router_medium_guard"].append(
            choose_video(
                sample_id,
                phase12_repeat_guard,
                "phase0_depth_phase12_repeat_router_medium_guard",
            )
        )

    for name, report_rows in variants.items():
        write_report(args.out_dir / f"{name}_report.csv", report_rows)
        counts: dict[str, int] = {}
        for report_row in report_rows:
            counts[report_row["chosen"]] = counts.get(report_row["chosen"], 0) + 1
        print(f"{name}: {counts}")
    print(f"out_dir={args.out_dir}")


if __name__ == "__main__":
    main()

"""Build a report by choosing candidate reports from manifest phase rules."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from wan_ticft.evaluation.evaluate_video_reports import parse_reports, read_report


def read_manifest(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--default", required=True)
    parser.add_argument("--phase_candidate", action="append", default=[], help="phase=candidate")
    parser.add_argument("--reports", nargs="+", required=True, help="candidate=/path/report.csv")
    args = parser.parse_args()

    report_paths = parse_reports(args.reports)
    reports = {label: read_report(path) for label, path in report_paths}
    phase_map = {}
    for item in args.phase_candidate:
        phase, candidate = item.split("=", 1)
        phase_map[str(phase)] = candidate
    if args.default not in reports:
        raise KeyError(f"default candidate missing from reports: {args.default}")
    for candidate in phase_map.values():
        if candidate not in reports:
            raise KeyError(f"phase candidate missing from reports: {candidate}")

    rows = []
    choices = []
    for row in read_manifest(args.manifest):
        sample_id = row["sample_id"]
        phase = str(row.get("phase", ""))
        candidate = phase_map.get(phase, args.default)
        video = reports[candidate].get(sample_id)
        if video is None:
            candidate = args.default
            video = reports[candidate].get(sample_id)
        if video is None:
            rows.append({"sample_id": sample_id, "video": "", "status": "missing", "chosen_candidate": candidate})
            continue
        rows.append({"sample_id": sample_id, "video": str(video), "status": "ok", "chosen_candidate": candidate})
        choices.append({"sample_id": sample_id, "phase": phase, "chosen_candidate": candidate, "video": str(video)})

    args.out_dir.mkdir(parents=True, exist_ok=True)
    with (args.out_dir / "report.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["sample_id", "video", "status", "chosen_candidate"])
        writer.writeheader()
        writer.writerows(rows)
    with (args.out_dir / "choices.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["sample_id", "phase", "chosen_candidate", "video"])
        writer.writeheader()
        writer.writerows(choices)
    counts: dict[str, int] = {}
    for row in choices:
        counts[row["chosen_candidate"]] = counts.get(row["chosen_candidate"], 0) + 1
    print(f"rows={len(rows)}")
    print(f"choices={len(choices)}")
    print(f"counts={counts}")
    print(f"report={args.out_dir / 'report.csv'}")


if __name__ == "__main__":
    main()

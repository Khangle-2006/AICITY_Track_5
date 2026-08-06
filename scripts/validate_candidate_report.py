from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarize and sanity-check a hybrid Track 5 candidate report CSV."
    )
    parser.add_argument(
        "report_csv",
        type=Path,
        nargs="?",
        default=Path(__file__).resolve().parents[1]
        / "reports"
        / "hybrid_multiframe_mf_h8_ltxhf_t55_wts_candidate_report.csv",
    )
    parser.add_argument("--expected-samples", type=int, default=71)
    parser.add_argument("--expected-frames", type=int, default=5194)
    args = parser.parse_args()

    with args.report_csv.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    sample_count = len(rows)
    frame_count = sum(int(row.get("frames_found") or 0) for row in rows)
    statuses = Counter(row.get("status", "") for row in rows)
    choices = Counter(row.get("source_choice", "") for row in rows)
    views = Counter(row.get("view_type", "") for row in rows)

    print(f"report={args.report_csv}")
    print(f"samples={sample_count}")
    print(f"frames={frame_count}")
    print(f"statuses={dict(statuses)}")
    print(f"source_choices={dict(choices)}")
    print(f"view_types={dict(views)}")

    problems = []
    if sample_count != args.expected_samples:
        problems.append(f"expected {args.expected_samples} samples, found {sample_count}")
    if frame_count != args.expected_frames:
        problems.append(f"expected {args.expected_frames} frames, found {frame_count}")
    if set(statuses) != {"ok"}:
        problems.append(f"unexpected status values: {dict(statuses)}")

    if problems:
        for problem in problems:
            print(f"ERROR: {problem}")
        raise SystemExit(1)

    print("ok")


if __name__ == "__main__":
    main()

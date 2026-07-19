"""Build a per-sample oracle report from existing candidate video reports."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from baseline.frame_io import import_cv2_np
from wan_ticft.evaluation.evaluate_video_reports import (
    evaluate_candidate,
    parse_reports,
    read_manifest,
    read_report,
)


def choose(rows: list[dict], metric: str) -> dict:
    if metric == "absdiff":
        return min(rows, key=lambda row: float(row["absdiff"]))
    return max(rows, key=lambda row: float(row[metric]))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--reports", nargs="+", required=True, help="label=/path/report.csv")
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--metric", choices=["psnr", "ssim", "absdiff"], default="absdiff")
    parser.add_argument("--offset_mode", choices=["index", "timestamp"], default="timestamp")
    parser.add_argument("--include_repeat_last", action="store_true")
    args = parser.parse_args()

    cv2, np = import_cv2_np()
    manifest = read_manifest(args.manifest)
    parsed_reports = parse_reports(args.reports)
    reports = {label: read_report(path) for label, path in parsed_reports}
    labels = [label for label, _ in parsed_reports]
    if args.include_repeat_last:
        labels.append("repeat_last")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    score_rows = []
    choice_rows = []
    report_rows = []
    for row in manifest:
        sample_id = row["sample_id"]
        candidates = []
        for label in labels:
            video = None if label == "repeat_last" else reports[label].get(sample_id)
            scored = evaluate_candidate(cv2, np, row, label, video, args.offset_mode)
            score_rows.append(scored)
            if scored["status"] == "ok" and label != "repeat_last":
                candidates.append(scored)
        if not candidates:
            continue
        winner = choose(candidates, args.metric)
        choice_rows.append(
            {
                "sample_id": sample_id,
                "winner": winner["candidate"],
                "metric": args.metric,
                "psnr": winner["psnr"],
                "ssim": winner["ssim"],
                "absdiff": winner["absdiff"],
                "video": winner["video"],
            }
        )
        report_rows.append(
            {
                "sample_id": sample_id,
                "video": winner["video"],
                "status": "ok",
                "chosen_candidate": winner["candidate"],
                "metric": args.metric,
                "psnr": winner["psnr"],
                "ssim": winner["ssim"],
                "absdiff": winner["absdiff"],
            }
        )

    if not report_rows:
        raise RuntimeError("no oracle choices written")

    with (args.out_dir / "oracle_report.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(report_rows[0]))
        writer.writeheader()
        writer.writerows(report_rows)
    with (args.out_dir / "oracle_choices.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(choice_rows[0]))
        writer.writeheader()
        writer.writerows(choice_rows)
    with (args.out_dir / "candidate_scores.csv").open("w", encoding="utf-8", newline="") as handle:
        fields = list(score_rows[0])
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(score_rows)

    counts: dict[str, int] = {}
    for row in choice_rows:
        counts[row["winner"]] = counts.get(row["winner"], 0) + 1
    print(f"rows={len(report_rows)}")
    print(f"metric={args.metric}")
    print(f"winner_counts={counts}")
    print(f"report={args.out_dir / 'oracle_report.csv'}")
    print(f"choices={args.out_dir / 'oracle_choices.csv'}")


if __name__ == "__main__":
    main()

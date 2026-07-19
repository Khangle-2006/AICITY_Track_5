"""Summarize candidate quality using macro-domain and worst-domain deltas."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path


METRICS = ("psnr", "ssim", "absdiff")


def mean(values):
    return sum(values) / max(1, len(values))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_csv", type=Path, required=True)
    parser.add_argument("--out_csv", type=Path, required=True)
    parser.add_argument("--baseline", default="repeat_last")
    parser.add_argument("--max_worst_psnr_drop", type=float, default=0.10)
    parser.add_argument("--max_worst_ssim_drop", type=float, default=0.002)
    parser.add_argument("--max_worst_absdiff_increase", type=float, default=0.20)
    args = parser.parse_args()

    with args.input_csv.open(encoding="utf-8", newline="") as handle:
        rows = [row for row in csv.DictReader(handle) if row.get("status", "ok") == "ok"]
    if not rows:
        raise RuntimeError("no valid rows")

    grouped = defaultdict(lambda: defaultdict(list))
    for row in rows:
        grouped[row["candidate"]][row["view_type"]].append(row)
    views = sorted({row["view_type"] for row in rows})
    if args.baseline not in grouped:
        raise RuntimeError(f"missing baseline candidate: {args.baseline}")

    view_means = {}
    for candidate, candidate_views in grouped.items():
        if any(view not in candidate_views for view in views):
            continue
        view_means[candidate] = {
            view: {
                metric: mean([float(row[metric]) for row in candidate_views[view]])
                for metric in METRICS
            }
            for view in views
        }

    baseline = view_means[args.baseline]
    output = []
    for candidate, values in view_means.items():
        psnr_deltas = [values[view]["psnr"] - baseline[view]["psnr"] for view in views]
        ssim_deltas = [values[view]["ssim"] - baseline[view]["ssim"] for view in views]
        absdiff_deltas = [values[view]["absdiff"] - baseline[view]["absdiff"] for view in views]
        macro_psnr = mean([values[view]["psnr"] for view in views])
        macro_ssim = mean([values[view]["ssim"] for view in views])
        macro_absdiff = mean([values[view]["absdiff"] for view in views])
        generalizes = (
            min(psnr_deltas) >= -args.max_worst_psnr_drop
            and min(ssim_deltas) >= -args.max_worst_ssim_drop
            and max(absdiff_deltas) <= args.max_worst_absdiff_increase
            and mean(psnr_deltas) > 0
            and mean(ssim_deltas) > 0
            and mean(absdiff_deltas) < 0
        )
        output.append(
            {
                "candidate": candidate,
                "domains": len(views),
                "macro_psnr": f"{macro_psnr:.6f}",
                "macro_ssim": f"{macro_ssim:.6f}",
                "macro_absdiff": f"{macro_absdiff:.6f}",
                "macro_d_psnr": f"{mean(psnr_deltas):+.6f}",
                "macro_d_ssim": f"{mean(ssim_deltas):+.6f}",
                "macro_d_absdiff": f"{mean(absdiff_deltas):+.6f}",
                "worst_d_psnr": f"{min(psnr_deltas):+.6f}",
                "worst_d_ssim": f"{min(ssim_deltas):+.6f}",
                "worst_d_absdiff": f"{max(absdiff_deltas):+.6f}",
                "generalizes": "yes" if generalizes else "no",
            }
        )

    output.sort(
        key=lambda row: (
            row["generalizes"] == "yes",
            float(row["macro_d_psnr"]),
            float(row["macro_d_ssim"]),
            -float(row["macro_d_absdiff"]),
        ),
        reverse=True,
    )
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.out_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=output[0].keys())
        writer.writeheader()
        writer.writerows(output)

    print(f"views: {views}")
    print(f"{'Candidate':<24} {'dPSNR':>9} {'dSSIM':>9} {'dAbs':>9} {'worstP':>9} {'worstS':>9} {'worstA':>9} {'Pass':>6}")
    print("-" * 94)
    for row in output:
        print(
            f"{row['candidate']:<24} {float(row['macro_d_psnr']):>+9.4f} "
            f"{float(row['macro_d_ssim']):>+9.4f} {float(row['macro_d_absdiff']):>+9.4f} "
            f"{float(row['worst_d_psnr']):>+9.4f} {float(row['worst_d_ssim']):>+9.4f} "
            f"{float(row['worst_d_absdiff']):>+9.4f} {row['generalizes']:>6}"
        )
    print(f"output {args.out_csv}")


if __name__ == "__main__":
    main()

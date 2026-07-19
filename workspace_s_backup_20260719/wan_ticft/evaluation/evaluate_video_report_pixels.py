from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

from baseline.frame_io import import_cv2_np
from baseline.metrics import mean_absdiff, psnr, ssim_gray
from wan_ticft.evaluation.audit_perceptual_video_distribution import load_full_video
from wan_ticft.evaluation.evaluate_video_reports import read_report


def parse_report_arg(value: str) -> tuple[str, Path]:
    if "=" not in value:
        path = Path(value)
        return path.stem, path
    name, raw_path = value.split("=", 1)
    return name, Path(raw_path)


def resize_like(cv2, pred, target):
    if pred.shape[:2] == target.shape[:2]:
        return pred
    return cv2.resize(pred, (target.shape[1], target.shape[0]), interpolation=cv2.INTER_CUBIC)


def score_sequence(cv2, np, preds, targets):
    count = min(len(preds), len(targets))
    if count <= 0:
        return None
    psnrs = []
    ssims = []
    diffs = []
    for pred, target in zip(preds[:count], targets[:count]):
        pred = resize_like(cv2, pred, target)
        psnrs.append(psnr(cv2, np, pred, target))
        ssims.append(ssim_gray(cv2, np, pred, target))
        diffs.append(mean_absdiff(np, pred, target))
    return float(np.mean(psnrs)), float(np.mean(ssims)), float(np.mean(diffs)), count


def read_report_rows(path: Path) -> dict[str, dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return {row["sample_id"]: row for row in csv.DictReader(handle) if row.get("status") == "ok"}


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate full-video report pixel metrics against target report.")
    parser.add_argument("--target_report", type=Path, required=True)
    parser.add_argument("--report", action="append", required=True, help="NAME=CSV")
    parser.add_argument("--out_csv", type=Path, required=True)
    args = parser.parse_args()

    cv2, np = import_cv2_np()
    target_paths = read_report(args.target_report)
    reports = [(name, path, read_report(path), read_report_rows(path)) for name, path in map(parse_report_arg, args.report)]

    rows = []
    for sample_id, target_path in sorted(target_paths.items()):
        targets = load_full_video(cv2, target_path)
        if not targets:
            continue
        for name, _, candidate_paths, candidate_rows in reports:
            candidate_path = candidate_paths.get(sample_id)
            if candidate_path is None:
                continue
            preds = load_full_video(cv2, candidate_path)
            values = score_sequence(cv2, np, preds, targets)
            if values is None:
                continue
            p, s, d, count = values
            meta = candidate_rows.get(sample_id, {})
            rows.append(
                {
                    "sample_id": sample_id,
                    "view_type": meta.get("view_type", ""),
                    "candidate": name,
                    "frames": count,
                    "psnr": f"{p:.6f}",
                    "ssim": f"{s:.6f}",
                    "absdiff": f"{d:.6f}",
                }
            )
            print(f"{sample_id} | {name} | PSNR={p:.4f} SSIM={s:.4f}", flush=True)

    if not rows:
        raise RuntimeError("no scored rows")
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.out_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    grouped = defaultdict(list)
    for row in rows:
        grouped[("Overall", row["candidate"])].append(row)
        if row["view_type"]:
            grouped[(row["view_type"], row["candidate"])].append(row)

    for domain in ["Overall", *sorted({row["view_type"] for row in rows if row["view_type"]})]:
        print()
        print(domain)
        print("-" * 64)
        print(f"{'Candidate':<28} {'N':>5} {'PSNR':>10} {'SSIM':>10} {'AbsDiff':>10}")
        print("-" * 64)
        candidates = [name for name, _, _, _ in reports]
        for candidate in candidates:
            items = grouped.get((domain, candidate), [])
            if not items:
                continue
            p = np.mean([float(item["psnr"]) for item in items])
            s = np.mean([float(item["ssim"]) for item in items])
            d = np.mean([float(item["absdiff"]) for item in items])
            print(f"{candidate:<28} {len(items):>5} {p:>10.4f} {s:>10.4f} {d:>10.4f}")
    print(f"output={args.out_csv}")


if __name__ == "__main__":
    main()

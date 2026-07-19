"""Measure future-flow warp and disocclusion-hole headroom across all domains."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from baseline.frame_io import import_cv2_np, read_rgb
from baseline.generators import compute_flow, warp_rgb
from baseline.metrics import mean_absdiff, psnr, ssim_gray
from wan_ticft.evaluation.evaluate_bdd_warp_inpaint_oracle import (
    composite,
    dilate_holes,
    forward_splat,
    resize_like,
)


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_images(cv2, paths: list[str]):
    return [image for path in paths if (image := read_rgb(cv2, Path(path))) is not None]


def score(cv2, np, predictions, targets):
    values = [
        (
            psnr(cv2, np, prediction, target),
            ssim_gray(cv2, np, prediction, target),
            mean_absdiff(np, prediction, target),
        )
        for prediction, target in zip(predictions, targets)
    ]
    return tuple(float(np.mean([value[index] for value in values])) for index in range(3))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out_csv", type=Path, required=True)
    parser.add_argument("--flow_width", type=int, default=640)
    parser.add_argument("--hole_dilate", type=int, nargs="+", default=[0, 4, 8])
    parser.add_argument("--limit_per_domain", type=int, default=0)
    args = parser.parse_args()

    cv2, np = import_cv2_np()
    rows_out = []
    domain_counts: dict[str, int] = {}
    for position, row in enumerate(read_jsonl(args.manifest), start=1):
        domain = str(row.get("view_type", "unknown"))
        if args.limit_per_domain and domain_counts.get(domain, 0) >= args.limit_per_domain:
            continue
        paths = row.get("frames") or {}
        conditions = load_images(cv2, paths.get("condition") or [])
        targets = load_images(cv2, paths.get("target") or [])
        if not conditions or not targets:
            print(f"[{position}] {row.get('sample_id', '')} | skipped", flush=True)
            continue

        last = conditions[-1]
        candidates = {
            "repeat_last": [],
            "oracle_backward_flow": [],
            **{f"oracle_splat_repeat_hole_d{radius}": [] for radius in args.hole_dilate},
            **{f"oracle_splat_target_hole_d{radius}": [] for radius in args.hole_dilate},
        }
        hole_fractions = {radius: [] for radius in args.hole_dilate}
        for target in targets:
            last_frame = resize_like(cv2, last, target)
            forward = compute_flow(cv2, np, last_frame, target, args.flow_width)
            splat, visible = forward_splat(np, last_frame, forward)
            backward = compute_flow(cv2, np, target, last_frame, args.flow_width)
            candidates["repeat_last"].append(last_frame)
            candidates["oracle_backward_flow"].append(warp_rgb(cv2, np, last_frame, backward, -1.0))
            for radius in args.hole_dilate:
                holes = dilate_holes(cv2, np, visible, radius)
                hole_fractions[radius].append(float(np.mean(holes)))
                candidates[f"oracle_splat_repeat_hole_d{radius}"].append(
                    composite(np, splat, last_frame, holes)
                )
                candidates[f"oracle_splat_target_hole_d{radius}"].append(
                    composite(np, splat, target, holes)
                )

        for candidate, predictions in candidates.items():
            p, s, d = score(cv2, np, predictions, targets)
            radius = next(
                (value for value in args.hole_dilate if candidate.endswith(f"_d{value}")),
                None,
            )
            rows_out.append(
                {
                    "sample_id": row["sample_id"],
                    "view_type": domain,
                    "phase": row.get("phase", ""),
                    "candidate": candidate,
                    "target_count": len(targets),
                    "hole_fraction": (
                        f"{np.mean(hole_fractions[radius]):.6f}" if radius is not None else ""
                    ),
                    "psnr": f"{p:.6f}",
                    "ssim": f"{s:.6f}",
                    "absdiff": f"{d:.6f}",
                    "status": "ok",
                }
            )
        domain_counts[domain] = domain_counts.get(domain, 0) + 1
        print(f"[{position}] {domain} | {row['sample_id']} | ok", flush=True)

    if not rows_out:
        raise RuntimeError("no samples evaluated")
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.out_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows_out[0].keys())
        writer.writeheader()
        writer.writerows(rows_out)

    print(f"domains: {domain_counts}")
    for domain in sorted(domain_counts):
        print(f"\n{domain}")
        print(f"{'Candidate':<34} {'N':>5} {'Hole%':>8} {'PSNR':>10} {'SSIM':>10} {'AbsDiff':>10}")
        print("-" * 83)
        for candidate in dict.fromkeys(row["candidate"] for row in rows_out):
            items = [
                row for row in rows_out
                if row["view_type"] == domain and row["candidate"] == candidate
            ]
            holes = [float(row["hole_fraction"]) for row in items if row["hole_fraction"]]
            print(
                f"{candidate:<34} {len(items):>5} "
                f"{100.0 * np.mean(holes) if holes else 0.0:>8.2f} "
                f"{np.mean([float(row['psnr']) for row in items]):>10.4f} "
                f"{np.mean([float(row['ssim']) for row in items]):>10.4f} "
                f"{np.mean([float(row['absdiff']) for row in items]):>10.4f}"
            )
    print(f"output {args.out_csv}")


if __name__ == "__main__":
    main()

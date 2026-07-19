from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path

from baseline.frame_io import import_cv2_np, numeric_pngs, read_rgb, write_rgb
from baseline.verify import verify_sample_output
from wan_ticft.evaluation.evaluate_bdd_ego_motion import region_predictions
from wan_ticft.evaluation.evaluate_ltx_region_lowfreq import low_frequency_correct


def main() -> None:
    parser = argparse.ArgumentParser(description="Apply low-frequency region correction to BDD test PNG outputs.")
    parser.add_argument("--test_csv", type=Path, required=True)
    parser.add_argument("--ltx_dir", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--sigma", type=float, default=16.0)
    parser.add_argument("--alpha", type=float, default=0.50)
    parser.add_argument("--estimate_width", type=int, default=640)
    parser.add_argument("--clean", action="store_true")
    args = parser.parse_args()

    cv2, np = import_cv2_np()
    with args.test_csv.open(encoding="utf-8", newline="") as handle:
        rows = [row for row in csv.DictReader(handle) if row.get("view_type") == "bdd"]

    if args.clean and args.out_dir.exists():
        shutil.rmtree(args.out_dir)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    report = []
    for position, row in enumerate(rows, start=1):
        output_name = row["source_folder"]
        target_n = int(row["target_N"])
        source_dir = args.ltx_dir / output_name
        out_sample = args.out_dir / output_name
        ltx_paths = numeric_pngs(source_dir)
        history_paths = numeric_pngs(Path(row["input_dir"]))
        if len(ltx_paths) != target_n:
            raise RuntimeError(f"{output_name}: LTX frames={len(ltx_paths)} expected={target_n}")
        if len(history_paths) < 2:
            raise RuntimeError(f"{output_name}: fewer than 2 history frames")

        prev = read_rgb(cv2, history_paths[-2])
        last = read_rgb(cv2, history_paths[-1])
        if prev is None or last is None:
            raise RuntimeError(f"{output_name}: cannot read last history frames")
        history_gap = max(1, int(history_paths[-1].stem) - int(history_paths[-2].stem))
        offsets = list(range(1, target_n + 1))
        region = region_predictions(cv2, np, prev, last, offsets, history_gap, args.estimate_width)

        if out_sample.exists():
            shutil.rmtree(out_sample)
        out_sample.mkdir(parents=True, exist_ok=True)

        for index, (ltx_path, region_frame) in enumerate(zip(ltx_paths, region)):
            ltx = read_rgb(cv2, ltx_path)
            if ltx is None:
                raise RuntimeError(f"{output_name}: cannot read {ltx_path}")
            if region_frame.shape[:2] != ltx.shape[:2]:
                region_frame = cv2.resize(
                    region_frame,
                    (ltx.shape[1], ltx.shape[0]),
                    interpolation=cv2.INTER_CUBIC,
                )
            corrected = low_frequency_correct(cv2, np, ltx, region_frame, args.sigma, args.alpha)
            if not write_rgb(cv2, out_sample / f"{index}.png", corrected):
                raise RuntimeError(f"{output_name}: cannot write frame {index}")

        verified = verify_sample_output(out_sample, target_n)
        if not verified.ok:
            raise RuntimeError(f"{output_name}: verification failed: {verified.notes}")
        report.append(
            {
                "output_name": output_name,
                "target_N": target_n,
                "frames_found": verified.frames_found,
                "sigma": args.sigma,
                "alpha": args.alpha,
                "status": "ok",
            }
        )
        print(f"[{position}/{len(rows)}] {output_name} | ok", flush=True)

    with (args.out_dir / "report.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=report[0].keys())
        writer.writeheader()
        writer.writerows(report)
    (args.out_dir / "summary.json").write_text(
        json.dumps(
            {
                "rows": len(report),
                "frames": sum(int(row["frames_found"]) for row in report),
                "sigma": args.sigma,
                "alpha": args.alpha,
                "status": "ok",
            },
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()

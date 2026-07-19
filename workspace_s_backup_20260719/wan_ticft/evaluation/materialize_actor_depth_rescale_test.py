"""Rescale an actor-aware depth test residual relative to its frozen base frames."""

from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path

import cv2
import numpy as np

from baseline.frame_io import numeric_pngs, read_rgb, write_rgb
from baseline.verify import verify_sample_output


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test_csv", type=Path, required=True)
    parser.add_argument("--base_dir", type=Path, required=True)
    parser.add_argument("--source_dir", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--ratio", type=float, required=True)
    parser.add_argument("--clean", action="store_true")
    args = parser.parse_args()

    if args.clean and args.out_dir.exists():
        shutil.rmtree(args.out_dir)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows = [row for row in read_csv(args.test_csv) if row.get("view_type") == "bdd"]
    reports = []
    for position, row in enumerate(rows, start=1):
        output_name = row["source_folder"]
        target_n = int(row["target_N"])
        base_paths = numeric_pngs(args.base_dir / output_name)
        source_paths = numeric_pngs(args.source_dir / output_name)
        if len(base_paths) != target_n or len(source_paths) != target_n:
            raise RuntimeError(
                f"{output_name}: base={len(base_paths)} source={len(source_paths)} expected={target_n}"
            )
        sample_out = args.out_dir / output_name
        sample_out.mkdir(parents=True, exist_ok=True)
        for index, (base_path, source_path) in enumerate(zip(base_paths, source_paths)):
            base = read_rgb(cv2, base_path)
            source = read_rgb(cv2, source_path)
            if base is None or source is None:
                raise RuntimeError(f"cannot read frame {output_name}/{index}")
            if source.shape != base.shape:
                source = cv2.resize(source, (base.shape[1], base.shape[0]), interpolation=cv2.INTER_AREA)
            output = base.astype(np.float32) + args.ratio * (source.astype(np.float32) - base.astype(np.float32))
            output = np.clip(output, 0, 255).astype(np.uint8)
            if not write_rgb(cv2, sample_out / f"{index}.png", output):
                raise RuntimeError(f"cannot write {sample_out / f'{index}.png'}")
        verified = verify_sample_output(sample_out, target_n)
        if not verified.ok:
            raise RuntimeError(f"{output_name}: verification failed: {verified.notes}")
        reports.append(
            {
                "output_name": output_name,
                "sample_id": row["sample_id"],
                "view_type": "bdd",
                "target_N": target_n,
                "frames_found": verified.frames_found,
                "residual_ratio": f"{args.ratio:.6f}",
                "status": "ok",
            }
        )
        print(f"[{position}/{len(rows)}] {output_name} | ok", flush=True)
    with (args.out_dir / "report.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(reports[0]))
        writer.writeheader()
        writer.writerows(reports)
    print(f"output={args.out_dir}")


if __name__ == "__main__":
    main()

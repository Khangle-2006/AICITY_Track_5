"""Link videos from one or more reports into labeled visual-QA directories."""

from __future__ import annotations

import argparse
import csv
import os
import shutil
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", action="append", required=True, help="LABEL=CSV")
    parser.add_argument("--view_type", required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--clean", action="store_true")
    args = parser.parse_args()

    if args.clean and args.out_dir.exists():
        shutil.rmtree(args.out_dir)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    counts = {}
    for value in args.report:
        label, raw_path = value.split("=", 1)
        destination = args.out_dir / label
        destination.mkdir(parents=True, exist_ok=True)
        count = 0
        with Path(raw_path).open("r", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                if row.get("view_type") != args.view_type or row.get("status") != "ok":
                    continue
                source = Path(row["video"])
                target = destination / f"{row['sample_id']}.mp4"
                if target.exists() or target.is_symlink():
                    target.unlink()
                os.symlink(source, target)
                count += 1
        counts[label] = count
    print(f"counts={counts}")
    print(f"out_dir={args.out_dir}")


if __name__ == "__main__":
    main()

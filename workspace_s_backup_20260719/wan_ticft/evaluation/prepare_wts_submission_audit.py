from __future__ import annotations

import argparse
import csv
import os
import shutil
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_csv", type=Path, required=True)
    parser.add_argument("--source_dir", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--clean", action="store_true")
    args = parser.parse_args()

    if args.clean and args.out_dir.exists():
        shutil.rmtree(args.out_dir)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    with args.test_csv.open(encoding="utf-8", newline="") as f:
        rows = [row for row in csv.DictReader(f) if row.get("view_type", "").startswith("wts_")]
    for row in rows:
        source = args.source_dir / row["source_folder"]
        destination = args.out_dir / row["source_folder"]
        if not source.exists():
            raise FileNotFoundError(source)
        try:
            os.symlink(source, destination, target_is_directory=True)
        except OSError:
            shutil.copytree(source, destination)
    print(f"WTS audit samples: {len(rows)}")


if __name__ == "__main__":
    main()

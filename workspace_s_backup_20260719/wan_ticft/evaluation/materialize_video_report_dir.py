from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Copy a video report into a flat video directory.")
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--clean", action="store_true")
    args = parser.parse_args()

    if args.clean and args.out_dir.exists():
        shutil.rmtree(args.out_dir)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    copied = 0
    with args.report.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("status") != "ok":
                continue
            sample_id = row.get("sample_id", "")
            source = Path(row.get("video", ""))
            if not sample_id or not source.exists():
                continue
            shutil.copy2(source, args.out_dir / f"{sample_id}{source.suffix}")
            copied += 1
    print(f"copied={copied} out_dir={args.out_dir}", flush=True)


if __name__ == "__main__":
    main()

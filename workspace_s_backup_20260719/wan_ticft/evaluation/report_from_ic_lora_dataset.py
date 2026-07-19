from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Build video reports from an IC-LoRA dataset/validation JSON.")
    parser.add_argument("--json_path", type=Path, required=True)
    parser.add_argument("--out_csv", type=Path, required=True)
    parser.add_argument("--path_key", default="reference_path")
    parser.add_argument("--root", type=Path, default=None)
    args = parser.parse_args()

    rows = json.loads(args.json_path.read_text(encoding="utf-8"))
    root = args.root or args.json_path.parent
    report = []
    for item in rows:
        sample_id = item.get("sample_id")
        raw = item.get(args.path_key)
        if not sample_id or not raw:
            continue
        path = Path(raw)
        if not path.is_absolute():
            path = root / path
        report.append({"sample_id": sample_id, "video": str(path), "status": "ok"})
    if not report:
        raise RuntimeError("no report rows")
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.out_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["sample_id", "video", "status"])
        writer.writeheader()
        writer.writerows(report)
    print(f"rows={len(report)}")
    print(f"report={args.out_csv}")


if __name__ == "__main__":
    main()

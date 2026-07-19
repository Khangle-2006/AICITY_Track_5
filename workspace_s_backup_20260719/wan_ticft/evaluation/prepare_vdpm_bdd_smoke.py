from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path


def read_scores(path: Path) -> dict[str, dict[str, float]]:
    scores: dict[str, dict[str, float]] = {}
    with path.open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            if row["candidate"] not in {"region", "ltx16"}:
                continue
            scores.setdefault(row["sample_id"], {})[row["candidate"]] = float(row["psnr"])
    return {sample: values for sample, values in scores.items() if len(values) == 2}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ceiling_csv", type=Path, required=True)
    parser.add_argument("--history_dir", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    args = parser.parse_args()

    scores = read_scores(args.ceiling_csv)
    ranked = sorted(
        (
            {
                "sample_id": sample_id,
                "region_psnr": values["region"],
                "ltx16_psnr": values["ltx16"],
                "region_minus_ltx16": values["region"] - values["ltx16"],
            }
            for sample_id, values in scores.items()
        ),
        key=lambda row: row["region_minus_ltx16"],
    )
    selected = ranked[:2] + ranked[-2:]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    video_dir = args.out_dir / "history_videos"
    video_dir.mkdir(parents=True, exist_ok=True)

    report = []
    for row in selected:
        source = args.history_dir / f"{row['sample_id']}.mp4"
        destination = video_dir / source.name
        if not source.exists():
            raise FileNotFoundError(source)
        shutil.copy2(source, destination)
        report.append(
            {
                **row,
                "selected_for": "ltx16_win" if row["region_minus_ltx16"] < 0 else "region_win",
                "history_video": str(destination),
            }
        )

    with (args.out_dir / "report.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=report[0].keys())
        writer.writeheader()
        writer.writerows(report)
    (args.out_dir / "summary.json").write_text(
        json.dumps({"rows": len(report), "selected": report}, indent=2), encoding="utf-8"
    )
    print(json.dumps({"rows": len(report), "selected": report}, indent=2))


if __name__ == "__main__":
    main()

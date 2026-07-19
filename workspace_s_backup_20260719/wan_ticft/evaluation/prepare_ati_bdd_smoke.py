from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from baseline.frame_io import import_cv2_np, numeric_pngs, read_rgb
from baseline.index_io import load_test_rows


def write_video(cv2, frames, path: Path, fps: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    height, width = frames[0].shape[:2]
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"cannot create {path}")
    for frame in frames:
        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    writer.release()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index_dir", type=Path, default=Path("/workspace/experiments/index"))
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=4)
    parser.add_argument("--fps", type=float, default=15.0)
    args = parser.parse_args()

    cv2, _ = import_cv2_np()
    rows = [row for row in load_test_rows(args.index_dir) if row.get("view_type") == "bdd"][: args.limit]
    if not rows:
        raise RuntimeError("no BDD test rows found")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "first_frames").mkdir(parents=True, exist_ok=True)
    report = []

    for row in rows:
        sample = row["source_folder"]
        paths = numeric_pngs(Path(row["input_dir"]))
        frames = [image for path in paths if (image := read_rgb(cv2, path)) is not None]
        status = "ok"
        notes = ""
        if len(frames) < 2:
            status = "bad"
            notes = "fewer than two readable history frames"
        else:
            write_video(cv2, frames, args.out_dir / "history_videos" / f"{sample}.mp4", args.fps)
            cv2.imwrite(
                str(args.out_dir / "first_frames" / f"{sample}.png"),
                cv2.cvtColor(frames[-1], cv2.COLOR_RGB2BGR),
            )
        report.append(
            {
                "sample": sample,
                "sample_id": row["sample_id"],
                "phase": row["phase"],
                "history_frames": len(frames),
                "target_N": row["target_N"],
                "caption_pedestrian": row.get("caption_pedestrian", ""),
                "caption_vehicle": row.get("caption_vehicle", ""),
                "status": status,
                "notes": notes,
            }
        )

    with (args.out_dir / "report.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=report[0].keys())
        writer.writeheader()
        writer.writerows(report)
    (args.out_dir / "summary.json").write_text(
        json.dumps(
            {
                "rows": len(report),
                "ok": sum(row["status"] == "ok" for row in report),
                "history_videos": str(args.out_dir / "history_videos"),
                "first_frames": str(args.out_dir / "first_frames"),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(json.dumps({"rows": len(report), "ok": sum(row["status"] == "ok" for row in report)}, indent=2))


if __name__ == "__main__":
    main()

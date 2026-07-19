from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from baseline.frame_io import import_cv2_np, numeric_pngs, read_rgb
from baseline.index_io import load_test_rows
from wan_ticft.evaluation.prepare_ati_bdd_smoke import write_video


def compatible_history_length(frame_count: int) -> int:
    """Return the largest LTX video-condition length of the form 8n+1."""
    if frame_count < 9:
        return 0
    return ((frame_count - 1) // 8) * 8 + 1


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

    history_dir = args.out_dir / "history_videos"
    history_dir.mkdir(parents=True, exist_ok=True)
    report = []

    for row in rows:
        sample = row["source_folder"]
        paths = numeric_pngs(Path(row["input_dir"]))
        frames = [image for path in paths if (image := read_rgb(cv2, path)) is not None]
        used_count = compatible_history_length(len(frames))
        status = "ok" if used_count else "bad"
        notes = "" if used_count else "LTX video extension requires at least 9 readable history frames"
        if used_count:
            write_video(cv2, frames[-used_count:], history_dir / f"{sample}.mp4", args.fps)

        report.append(
            {
                "sample": sample,
                "sample_id": row["sample_id"],
                "phase": row["phase"],
                "history_frames_available": len(frames),
                "history_frames_used": used_count,
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
                "history_videos": str(history_dir),
                "history_rule": "last 8n+1 readable frames",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(json.dumps({"rows": len(report), "ok": sum(row["status"] == "ok" for row in report)}, indent=2))


if __name__ == "__main__":
    main()

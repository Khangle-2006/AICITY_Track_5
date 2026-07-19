"""Convert BDD frame folders in a Track 5 submission into preview videos."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from baseline.frame_io import import_cv2_np, numeric_pngs


def bdd_output_names(index_csv: Path) -> list[tuple[str, str, int]]:
    rows = []
    with index_csv.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("view_type") != "bdd":
                continue
            output_name = row.get("output_name") or row.get("source_folder") or ""
            if not output_name:
                continue
            rows.append((output_name, row.get("sample_id", output_name), int(row.get("target_N") or 0)))
    return rows


def write_video(cv2, frame_paths: list[Path], out_path: Path, fps: float, codec: str) -> None:
    first = cv2.imread(str(frame_paths[0]), cv2.IMREAD_COLOR)
    if first is None:
        raise RuntimeError(f"cannot read {frame_paths[0]}")
    height, width = first.shape[:2]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*codec), fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"cannot create {out_path}")
    writer.write(first)
    for path in frame_paths[1:]:
        frame = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if frame is None:
            writer.release()
            raise RuntimeError(f"cannot read {path}")
        if frame.shape[:2] != (height, width):
            frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
        writer.write(frame)
    writer.release()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index_csv", type=Path, default=Path("/workspace/experiments/index/test_index.csv"))
    parser.add_argument("--submission_dir", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument("--codec", default="mp4v")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    cv2, _ = import_cv2_np()
    rows = bdd_output_names(args.index_csv)
    if args.limit:
        rows = rows[: args.limit]
    if not rows:
        raise RuntimeError(f"no BDD rows found in {args.index_csv}")

    report = []
    for position, (output_name, sample_id, target_n) in enumerate(rows, start=1):
        frame_dir = args.submission_dir / output_name
        frame_paths = numeric_pngs(frame_dir)
        status = "ok"
        notes = ""
        video_path = args.out_dir / f"{output_name}.mp4"
        try:
            if not frame_paths:
                raise RuntimeError("no png frames")
            if target_n and len(frame_paths) != target_n:
                notes = f"frame count {len(frame_paths)} != target_N {target_n}"
            write_video(cv2, frame_paths, video_path, args.fps, args.codec)
        except Exception as exc:
            status = "bad"
            notes = str(exc)
            video_path = Path("")
        report.append(
            {
                "output_name": output_name,
                "sample_id": sample_id,
                "target_N": target_n,
                "frames": len(frame_paths),
                "video": str(video_path),
                "status": status,
                "notes": notes,
            }
        )
        print(f"[{position}/{len(rows)}] {output_name} | {status} frames={len(frame_paths)}", flush=True)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    report_path = args.out_dir / "report.csv"
    with report_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=report[0].keys())
        writer.writeheader()
        writer.writerows(report)
    ok = sum(1 for row in report if row["status"] == "ok")
    print(f"videos={args.out_dir}")
    print(f"report={report_path}")
    print(f"ok={ok}/{len(report)}")


if __name__ == "__main__":
    main()

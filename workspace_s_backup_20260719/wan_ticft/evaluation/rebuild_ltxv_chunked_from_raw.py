from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from baseline.frame_io import import_cv2_np
from wan_ticft.evaluation.run_ltxv_val_chunked_gate import (
    compatible_history_length,
    compatible_total_length,
    read_video_bgr,
    target_horizon,
    write_report,
    write_video,
)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def read_manifest(path: Path) -> dict[str, dict]:
    rows = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                rows[row["sample_id"]] = row
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Rebuild fixed chunked LTX videos from existing raw chunk outputs.")
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--max_history_frames", type=int, default=41)
    parser.add_argument("--width", type=int, default=1216)
    parser.add_argument("--height", type=int, default=704)
    args = parser.parse_args()

    cv2, _ = import_cv2_np()
    report_rows = read_csv(args.out_dir / "report.csv")
    manifest = read_manifest(args.out_dir / "selected_manifest.jsonl")
    rebuilt_rows = []
    for row in report_rows:
        sample_id = row["sample_id"]
        manifest_row = manifest[sample_id]
        history_count = int(float(row["history_frames"]))
        source_fps = float(row["source_fps"])
        chunk_frames = int(float(row["chunk_frames"]))
        wanted_future = target_horizon(manifest_row, chunk_frames * 3)
        generated_video = []
        generated_future_count = 0
        chunk_index = 0
        while generated_future_count < wanted_future:
            chunk_index += 1
            remaining = wanted_future - generated_future_count
            chunk = min(chunk_frames, remaining)
            current_history = compatible_history_length(history_count + generated_future_count, args.max_history_frames)
            expected_total = compatible_total_length(current_history + chunk)
            raw_dir = args.out_dir / "raw" / sample_id / f"chunk{chunk_index:02d}"
            raw_videos = list(raw_dir.glob("*.mp4"))
            if len(raw_videos) != 1:
                raise RuntimeError(f"{sample_id}: expected one raw video in {raw_dir}, found {len(raw_videos)}")
            output_frames, _ = read_video_bgr(cv2, raw_videos[0])
            if len(output_frames) < expected_total:
                raise RuntimeError(
                    f"{sample_id}: chunk {chunk_index} has {len(output_frames)} frames, "
                    f"expected at least {expected_total}; rerun this chunk with the fixed generator"
                )
            segment = output_frames[current_history - 1 : current_history + chunk]
            if len(segment) != chunk + 1:
                raise RuntimeError(f"{sample_id}: chunk {chunk_index} decoded {len(segment)}/{chunk + 1}")
            if not generated_video:
                generated_video.extend(segment)
            else:
                generated_video.extend(segment[1:])
            generated_future_count += chunk

        future_path = args.out_dir / "videos" / f"{sample_id}.mp4"
        write_video(cv2, generated_video, future_path, source_fps, args.width, args.height)
        row["future_frames"] = len(generated_video)
        row["chunks"] = chunk_index
        rebuilt_rows.append(row)
        print(f"{sample_id}: rebuilt {len(generated_video)} frames", flush=True)

    write_report(args.out_dir / "report.csv", rebuilt_rows)
    print(f"report={args.out_dir / 'report.csv'}")


if __name__ == "__main__":
    main()

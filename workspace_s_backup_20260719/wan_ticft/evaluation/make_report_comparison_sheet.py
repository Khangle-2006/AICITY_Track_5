from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from baseline.frame_io import import_cv2_np, read_rgb
from wan_ticft.evaluation.evaluate_video_reports import map_target_offsets, read_report, read_video_rgb_frames, video_fps


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def parse_reports(items: list[str]) -> dict[str, Path]:
    out = {}
    for item in items:
        name, path = item.split("=", 1)
        out[name] = Path(path)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--reports", nargs="+", required=True)
    parser.add_argument("--sample_id", action="append", required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--frames", type=int, default=8)
    parser.add_argument("--tile_width", type=int, default=240)
    args = parser.parse_args()

    cv2, np = import_cv2_np()
    rows = {row["sample_id"]: row for row in read_jsonl(args.manifest)}
    reports = {name: read_report(path) for name, path in parse_reports(args.reports).items()}
    args.out_dir.mkdir(parents=True, exist_ok=True)
    index_rows = []

    for sample_id in args.sample_id:
        row = rows[sample_id]
        target_paths = (row.get("frames") or {}).get("target") or []
        if not target_paths:
            continue
        count = min(args.frames, len(target_paths))
        positions = sorted(set(np.linspace(0, len(target_paths) - 1, count, dtype=int).tolist()))
        target_frames = [read_rgb(cv2, Path(target_paths[pos])) for pos in positions]
        grid_rows = [("target", target_frames)]
        for name, report in reports.items():
            video = report.get(sample_id)
            offsets = map_target_offsets(row, video_fps(cv2, video))
            frame_map = read_video_rgb_frames(cv2, video, offsets)
            frames = [frame_map.get(offsets[pos]) for pos in positions]
            grid_rows.append((name, frames))

        reference = next(frame for frame in target_frames if frame is not None)
        ref_h, ref_w = reference.shape[:2]
        tile_h = int(round(args.tile_width * ref_h / ref_w))
        resized_rows = []
        for label, frames in grid_rows:
            tiles = []
            for pos, frame in zip(positions, frames):
                if frame is None:
                    tile = np.zeros((tile_h, args.tile_width, 3), dtype=np.uint8)
                    cv2.putText(tile, "missing", (8, tile_h // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
                else:
                    tile = cv2.resize(frame, (args.tile_width, tile_h), interpolation=cv2.INTER_AREA)
                cv2.putText(tile, f"{label} f{pos}", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.putText(tile, f"{label} f{pos}", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)
                tiles.append(tile)
            resized_rows.append(np.concatenate(tiles, axis=1))
        sheet = np.concatenate(resized_rows, axis=0)
        out = args.out_dir / f"{sample_id}.jpg"
        cv2.imwrite(str(out), cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR))
        index_rows.append({"sample_id": sample_id, "sheet": str(out)})
        print(f"ok {out}")

    with (args.out_dir / "report.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["sample_id", "sheet"])
        writer.writeheader()
        writer.writerows(index_rows)


if __name__ == "__main__":
    main()

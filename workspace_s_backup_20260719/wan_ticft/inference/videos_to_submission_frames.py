"""Convert Wan2.2 generated videos into Track 5 frame folders.

The output layout matches ``baseline.submission`` expectations:

    gen_dir/<output_name>/<000000.png ...>

``output_name`` is resolved from the test index using the same source-folder
collision rule as the baseline package.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path
from typing import Any


def print_header(title: str) -> None:
    print("\n" + "=" * 100)
    print(title)
    print("=" * 100)


def print_kv(key: str, value: object, width: int = 42) -> None:
    print(f"{key:<{width}} {value}")


def parse_int(value: str | int | float | None, default: int = 0) -> int:
    try:
        if value is None or value == "":
            return default
        return int(float(value))
    except Exception:
        return default


def safe_name(value: str) -> str:
    keep = []
    for ch in value:
        if ch.isalnum() or ch in ("-", "_", "."):
            keep.append(ch)
        else:
            keep.append("_")
    out = "".join(keep).strip("._")
    return out or "sample"


def resolve_output_name(row: dict[str, str], used: set[str]) -> str:
    base = safe_name(row.get("source_folder") or row.get("sample_id") or "sample")
    if base not in used:
        used.add(base)
        return base

    fallback = safe_name(row.get("sample_id") or base)
    if fallback not in used:
        used.add(fallback)
        return fallback

    i = 2
    while f"{fallback}_{i}" in used:
        i += 1
    name = f"{fallback}_{i}"
    used.add(name)
    return name


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def build_expected(index_dir: Path) -> list[dict[str, Any]]:
    rows = read_csv(index_dir / "test_index.csv")
    used: set[str] = set()
    expected: list[dict[str, Any]] = []
    for row in rows:
        expected.append(
            {
                "sample_id": row.get("sample_id", ""),
                "source_folder": row.get("source_folder", ""),
                "dataset_part": row.get("dataset_part", ""),
                "view_type": row.get("view_type", ""),
                "target_N": parse_int(row.get("target_N"), 0),
                "width": parse_int(row.get("frame_width"), 1280),
                "height": parse_int(row.get("frame_height"), 720),
                "output_name": resolve_output_name(row, used),
            }
        )
    return expected


def build_video_map(report_csv: Path) -> dict[str, Path]:
    rows = read_csv(report_csv)
    video_map: dict[str, Path] = {}
    for row in rows:
        if row.get("status") != "ok":
            continue
        sample_id = row.get("sample_id", "")
        video = Path(row.get("video", ""))
        if sample_id and video.exists():
            video_map[sample_id] = video
    return video_map


def extract_video_frames(
    *,
    video_path: Path,
    out_dir: Path,
    count: int,
    width: int,
    height: int,
    jpg_quality: int = 95,
) -> tuple[int, str]:
    try:
        import cv2
    except Exception as exc:
        return 0, f"opencv import failed: {exc}"

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return 0, f"could not open video: {video_path}"

    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    written = 0
    note = ""
    for idx in range(count):
        ok, frame = cap.read()
        if not ok:
            note = f"video ended at frame {idx}, expected {count}"
            break
        if width > 0 and height > 0:
            h, w = frame.shape[:2]
            if w != width or h != height:
                frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_CUBIC)
        out_path = out_dir / f"{idx:06d}.png"
        ok_write = cv2.imwrite(str(out_path), frame, [cv2.IMWRITE_PNG_COMPRESSION, 3])
        if not ok_write:
            note = f"failed writing {out_path}"
            break
        written += 1

    cap.release()
    _ = jpg_quality
    return written, note


def write_report(path: Path, rows: list[dict[str, Any]]) -> None:
    keys = [
        "output_name",
        "sample_id",
        "source_folder",
        "dataset_part",
        "view_type",
        "target_N",
        "width",
        "height",
        "video",
        "output_dir",
        "frames_written",
        "status",
        "notes",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in keys})


def write_summary(path: Path, rows: list[dict[str, Any]]) -> None:
    by_status: dict[str, int] = {}
    by_view: dict[str, int] = {}
    frames = 0
    for row in rows:
        by_status[row["status"]] = by_status.get(row["status"], 0) + 1
        view = row.get("view_type", "")
        by_view[view] = by_view.get(view, 0) + 1
        frames += parse_int(row.get("frames_written"), 0)
    summary = {
        "rows": len(rows),
        "frames_written": frames,
        "by_status": by_status,
        "by_view": by_view,
    }
    path.write_text(json.dumps(summary, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index_dir", type=Path, required=True)
    parser.add_argument("--wan_report_csv", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--report_csv", type=Path, default=None)
    parser.add_argument("--clean", action="store_true", default=True)
    parser.add_argument("--no_clean", action="store_false", dest="clean")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.index_dir = args.index_dir.resolve()
    args.wan_report_csv = args.wan_report_csv.resolve()
    args.out_dir = args.out_dir.resolve()
    report_csv = args.report_csv.resolve() if args.report_csv else args.out_dir.parent / f"{args.out_dir.name}_verify.csv"

    print_header("WAN VIDEOS TO TRACK 5 FRAMES")
    print_kv("index_dir", args.index_dir)
    print_kv("wan_report_csv", args.wan_report_csv)
    print_kv("out_dir", args.out_dir)
    print_kv("report_csv", report_csv)

    if args.clean and args.out_dir.exists():
        shutil.rmtree(args.out_dir)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    expected = build_expected(args.index_dir)
    video_map = build_video_map(args.wan_report_csv)

    results: list[dict[str, Any]] = []
    for item in expected:
        sample_id = item["sample_id"]
        video = video_map.get(sample_id)
        out_sample_dir = args.out_dir / item["output_name"]
        if video is None:
            row = {
                **item,
                "video": "",
                "output_dir": str(out_sample_dir),
                "frames_written": 0,
                "status": "bad",
                "notes": "missing generated video in Wan report",
            }
        else:
            written, note = extract_video_frames(
                video_path=video,
                out_dir=out_sample_dir,
                count=item["target_N"],
                width=item["width"],
                height=item["height"],
            )
            ok = written == item["target_N"] and not note
            row = {
                **item,
                "video": str(video),
                "output_dir": str(out_sample_dir),
                "frames_written": written,
                "status": "ok" if ok else "bad",
                "notes": note,
            }
        results.append(row)

    write_report(report_csv, results)
    write_summary(args.out_dir / "summary.json", results)

    bad = [r for r in results if r["status"] != "ok"]
    print_header("RESULT")
    print_kv("rows", len(results))
    print_kv("bad", len(bad))
    print_kv("frames_written", sum(parse_int(r.get("frames_written"), 0) for r in results))
    print_kv("report_csv", report_csv)
    print_kv("summary", args.out_dir / "summary.json")
    if bad:
        for row in bad[:20]:
            print(f"  - {row['sample_id']} | {row['notes']}")
        raise SystemExit(2)


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

from .console import kv, section
from .frame_io import import_cv2_np, numeric_pngs


def discover_inputs(root: Path, mode: str) -> list[Path]:
    if mode == "videos":
        if root.is_file() and root.suffix.lower() in {".mp4", ".avi", ".mov", ".mkv"}:
            return [root]
        return sorted(p for p in root.glob("*.mp4") if p.is_file())

    if numeric_pngs(root):
        return [root]
    return sorted(p for p in root.iterdir() if p.is_dir() and numeric_pngs(p) and not p.name.startswith("_"))


def sample_indices(total: int, count: int) -> list[int]:
    if total <= 0:
        return []
    if count <= 1:
        return [0]
    if total <= count:
        return list(range(total))
    return sorted({round(i * (total - 1) / (count - 1)) for i in range(count)})


def read_video_samples(cv2, video_path: Path, count: int) -> tuple[list[Any], dict[str, Any]]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return [], {"status": "failed", "notes": "cannot open video"}

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)

    frames = []
    failed = 0
    for idx in sample_indices(total, count):
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if ok and frame is not None:
            frames.append((idx, frame))
        else:
            failed += 1
    cap.release()

    return frames, {
        "status": "ok" if frames else "failed",
        "frames": total,
        "fps": fps,
        "width": width,
        "height": height,
        "failed_samples": failed,
        "notes": "",
    }


def read_frame_dir_samples(cv2, frame_dir: Path, count: int) -> tuple[list[Any], dict[str, Any]]:
    paths = numeric_pngs(frame_dir)
    frames = []
    failed = 0
    for idx in sample_indices(len(paths), count):
        frame = cv2.imread(str(paths[idx]), cv2.IMREAD_COLOR)
        if frame is not None:
            frames.append((idx, frame))
        else:
            failed += 1

    h, w = frames[0][1].shape[:2] if frames else (0, 0)
    return frames, {
        "status": "ok" if frames else "failed",
        "frames": len(paths),
        "fps": "",
        "width": w,
        "height": h,
        "failed_samples": failed,
        "notes": "",
    }


def label_tile(cv2, frame, label: str, tile_width: int):
    h, w = frame.shape[:2]
    scale = tile_width / max(w, 1)
    tile_height = max(1, int(round(h * scale)))
    tile = cv2.resize(frame, (tile_width, tile_height), interpolation=cv2.INTER_AREA)
    cv2.rectangle(tile, (0, 0), (tile_width, 28), (0, 0, 0), thickness=-1)
    cv2.putText(tile, label, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return tile


def make_contact_sheet(cv2, np, samples: list[Any], out_path: Path, tile_width: int, cols: int) -> bool:
    if not samples:
        return False
    tiles = [label_tile(cv2, frame, f"frame {idx}", tile_width) for idx, frame in samples]
    tile_h = max(t.shape[0] for t in tiles)
    tile_w = tile_width
    rows = (len(tiles) + cols - 1) // cols
    sheet = np.zeros((rows * tile_h, cols * tile_w, 3), dtype=np.uint8)

    for i, tile in enumerate(tiles):
        r = i // cols
        c = i % cols
        y = r * tile_h
        x = c * tile_w
        sheet[y : y + tile.shape[0], x : x + tile.shape[1]] = tile

    out_path.parent.mkdir(parents=True, exist_ok=True)
    return bool(cv2.imwrite(str(out_path), sheet))


def write_index_html(out_dir: Path, rows: list[dict[str, Any]]) -> None:
    lines = [
        "<!doctype html><html><head><meta charset='utf-8'>",
        "<style>body{font-family:Arial,sans-serif;margin:24px;background:#111;color:#eee}"
        ".item{margin:0 0 28px}.meta{font-size:13px;color:#bbb;margin:4px 0 8px}"
        "img{max-width:100%;height:auto;border:1px solid #333}</style>",
        "</head><body><h1>Video Contact Sheets</h1>",
    ]
    for row in rows:
        if row["status"] != "ok":
            continue
        rel = Path(row["sheet"]).name
        lines.append("<div class='item'>")
        lines.append(f"<h3>{row['sample_id']}</h3>")
        lines.append(
            f"<div class='meta'>frames={row['frames']} fps={row['fps']} "
            f"size={row['width']}x{row['height']} failed_samples={row['failed_samples']}</div>"
        )
        lines.append(f"<img src='{rel}' alt='{row['sample_id']}'>")
        lines.append("</div>")
    lines.append("</body></html>")
    (out_dir / "index.html").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Create visual contact sheets from generated videos or frame folders.")
    parser.add_argument("--input", required=True, type=Path, help="MP4 file, video directory, frame directory, or generated root.")
    parser.add_argument("--out_dir", required=True, type=Path)
    parser.add_argument("--mode", choices=["videos", "frames"], default="videos")
    parser.add_argument("--samples", type=int, default=8)
    parser.add_argument("--tile_width", type=int, default=320)
    parser.add_argument("--cols", type=int, default=4)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--report_csv", type=Path, default=None)
    args = parser.parse_args()

    cv2, np = import_cv2_np()
    inputs = discover_inputs(args.input, args.mode)
    if args.limit > 0:
        inputs = inputs[: args.limit]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    report_csv = args.report_csv or (args.out_dir / "contact_sheet_report.csv")

    section("VIDEO CONTACT SHEETS")
    kv("input", args.input)
    kv("out_dir", args.out_dir)
    kv("mode", args.mode)
    kv("inputs", len(inputs))
    kv("samples per input", args.samples)

    rows: list[dict[str, Any]] = []
    for path in inputs:
        sample_id = path.stem if path.is_file() else path.name
        if args.mode == "videos":
            samples, meta = read_video_samples(cv2, path, args.samples)
        else:
            samples, meta = read_frame_dir_samples(cv2, path, args.samples)

        out_path = args.out_dir / f"{sample_id}.jpg"
        ok = make_contact_sheet(cv2, np, samples, out_path, args.tile_width, args.cols)
        row = {
            "sample_id": sample_id,
            "input": str(path),
            "sheet": str(out_path),
            **meta,
        }
        if not ok:
            row["status"] = "failed"
            row["notes"] = row.get("notes", "") or "could not write sheet"
        rows.append(row)

    with report_csv.open("w", newline="", encoding="utf-8") as f:
        fieldnames = ["sample_id", "input", "sheet", "status", "frames", "fps", "width", "height", "failed_samples", "notes"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    write_index_html(args.out_dir, rows)

    section("RESULT")
    kv("report_csv", report_csv)
    kv("sheets ok", sum(1 for r in rows if r["status"] == "ok"))
    kv("failures", sum(1 for r in rows if r["status"] != "ok"))
    kv("index_html", args.out_dir / "index.html")


if __name__ == "__main__":
    main()

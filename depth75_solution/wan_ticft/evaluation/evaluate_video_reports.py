"""Evaluate generated video reports against target frames in a validation manifest.

Each generated video is sampled at the frame offsets recorded by the manifest.
This matches the frame-indexed submission contract and scores candidates using
real target frames.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from baseline.frame_io import import_cv2_np, read_rgb
from baseline.metrics import mean_absdiff, psnr, ssim_gray


FIELDS = [
    "sample_id",
    "dataset_part",
    "view_type",
    "phase",
    "candidate",
    "video",
    "source_fps",
    "video_fps",
    "target_count",
    "psnr",
    "ssim",
    "absdiff",
    "status",
    "notes",
]


def section(title: str) -> None:
    print("\n" + "=" * 100)
    print(title)
    print("=" * 100)


def kv(key: str, value: object, width: int = 42) -> None:
    print(f"{key:<{width}} {value}")


def read_manifest(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON: {exc}") from exc
    return rows


def read_report(path: Path) -> dict[str, Path]:
    out: dict[str, Path] = {}
    with path.open("r", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("status") not in {"ok", "skipped"}:
                continue
            sample_id = row.get("sample_id", "")
            video = Path(row.get("video", ""))
            if sample_id and video.exists():
                out[sample_id] = video
    return out


def parse_reports(values: list[str]) -> list[tuple[str, Path]]:
    reports = []
    for value in values:
        if "=" in value:
            label, raw_path = value.split("=", 1)
        else:
            raw_path = value
            label = Path(value).parent.name
        path = Path(raw_path).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(path)
        reports.append((label.strip(), path))
    return reports


def video_fps(cv2, video_path: Path) -> float:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return 0.0
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    cap.release()
    return fps


def read_video_rgb_frames(cv2, video_path: Path, indices: list[int]) -> dict[int, Any]:
    wanted = sorted(set(max(0, i) for i in indices))
    frames: dict[int, Any] = {}
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return frames
    # Sequential decode is dramatically faster for short/dense index sets and
    # lossless codecs such as FFV1, where random seeking repeatedly decodes
    # from the previous keyframe.
    if wanted and max(wanted) <= 512:
        wanted_set = set(wanted)
        for index in range(max(wanted) + 1):
            ok, bgr = cap.read()
            if not ok or bgr is None:
                break
            if index in wanted_set:
                frames[index] = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        cap.release()
        return frames
    for index in wanted:
        cap.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, bgr = cap.read()
        if ok and bgr is not None:
            frames[index] = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    cap.release()
    return frames


def resize_like(cv2, pred, target):
    th, tw = target.shape[:2]
    if pred.shape[:2] == (th, tw):
        return pred
    return cv2.resize(pred, (tw, th), interpolation=cv2.INTER_CUBIC)


def target_offsets(row: dict[str, Any]) -> list[int]:
    indices = row.get("frame_indices") or {}
    condition = indices.get("condition") or []
    target = indices.get("target") or []
    if not condition:
        return list(range(1, len(target) + 1))
    last_condition = int(condition[-1])
    return [max(1, int(index) - last_condition) for index in target]


def source_fps(row: dict[str, Any]) -> float:
    indices = row.get("frame_indices") or {}
    try:
        return float(indices.get("fps") or 0.0)
    except (TypeError, ValueError):
        return 0.0


def map_target_offsets(row: dict[str, Any], generated_fps: float) -> list[int]:
    offsets = target_offsets(row)
    src_fps = source_fps(row)
    if src_fps <= 0 or generated_fps <= 0:
        return offsets
    return [max(1, int(round(offset * generated_fps / src_fps))) for offset in offsets]


def load_targets(cv2, row: dict[str, Any]) -> list[Any]:
    paths = ((row.get("frames") or {}).get("target") or [])
    return [img for path in paths if (img := read_rgb(cv2, Path(path))) is not None]


def load_last_condition(cv2, row: dict[str, Any]):
    paths = ((row.get("frames") or {}).get("condition") or [])
    return read_rgb(cv2, Path(paths[-1])) if paths else None


def score_frames(cv2, np, preds: list[Any], targets: list[Any]) -> tuple[float, float, float]:
    psnrs = []
    ssims = []
    diffs = []
    for pred, target in zip(preds, targets):
        pred = resize_like(cv2, pred, target)
        psnrs.append(psnr(cv2, np, pred, target))
        ssims.append(ssim_gray(cv2, np, pred, target))
        diffs.append(mean_absdiff(np, pred, target))
    return float(np.mean(psnrs)), float(np.mean(ssims)), float(np.mean(diffs))


def evaluate_candidate(
    cv2,
    np,
    row: dict[str, Any],
    label: str,
    video: Path | None,
    offset_mode: str,
) -> dict[str, Any]:
    sample_id = str(row.get("sample_id", ""))
    targets = load_targets(cv2, row)
    src_fps = source_fps(row)
    gen_fps = video_fps(cv2, video) if video is not None else 0.0
    offsets = (
        map_target_offsets(row, gen_fps)
        if offset_mode == "timestamp" and video is not None
        else target_offsets(row)
    )
    notes = ""

    if not targets:
        preds = []
        notes = "no readable target frames"
    elif label == "repeat_last":
        last = load_last_condition(cv2, row)
        preds = [last] * len(targets) if last is not None else []
        if last is None:
            notes = "no readable condition frame"
    elif video is None:
        preds = []
        notes = "sample missing from report"
    else:
        frame_map = read_video_rgb_frames(cv2, video, offsets)
        preds = [frame_map[index] for index in offsets if index in frame_map]
        if len(preds) != len(targets):
            notes = f"read {len(preds)}/{len(targets)} generated frames"
        elif src_fps > 0 and gen_fps > 0:
            notes = f"frame-index mapped; source={src_fps:.3f} fps video={gen_fps:.3f} fps"

    status = "ok" if len(preds) == len(targets) and targets else "bad"
    p, s, d = score_frames(cv2, np, preds, targets) if status == "ok" else (0.0, 0.0, 0.0)
    return {
        "sample_id": sample_id,
        "dataset_part": row.get("dataset_part", ""),
        "view_type": row.get("view_type", ""),
        "phase": row.get("phase", ""),
        "candidate": label,
        "video": str(video or ""),
        "source_fps": f"{src_fps:.6f}" if src_fps > 0 else "",
        "video_fps": f"{gen_fps:.6f}" if gen_fps > 0 else "",
        "target_count": len(targets),
        "psnr": f"{p:.6f}" if status == "ok" else "",
        "ssim": f"{s:.6f}" if status == "ok" else "",
        "absdiff": f"{d:.6f}" if status == "ok" else "",
        "status": status,
        "notes": notes,
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows({key: row.get(key, "") for key in FIELDS} for row in rows)


def print_summary(np, rows: list[dict[str, Any]]) -> None:
    labels = sorted({row["candidate"] for row in rows})
    print(f"{'Candidate':<28} {'N':>5} {'PSNR':>10} {'SSIM':>10} {'AbsDiff':>10} {'Wins':>7}")
    print("-" * 76)
    winners: dict[str, int] = {}
    by_sample: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        if row["status"] == "ok":
            by_sample.setdefault(row["sample_id"], []).append(row)
    for items in by_sample.values():
        winner = max(items, key=lambda item: float(item["psnr"]))
        winners[winner["candidate"]] = winners.get(winner["candidate"], 0) + 1
    for label in labels:
        items = [row for row in rows if row["candidate"] == label and row["status"] == "ok"]
        if not items:
            continue
        p = float(np.mean([float(row["psnr"]) for row in items]))
        s = float(np.mean([float(row["ssim"]) for row in items]))
        d = float(np.mean([float(row["absdiff"]) for row in items]))
        print(f"{label:<28} {len(items):>5} {p:>10.4f} {s:>10.4f} {d:>10.4f} {winners.get(label, 0):>7}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--reports", nargs="+", required=True, help="Use label=/path/to/report.csv.")
    parser.add_argument("--out_csv", type=Path, required=True)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--include_repeat_last", action="store_true")
    parser.add_argument(
        "--offset_mode",
        choices=["frame", "timestamp"],
        default="frame",
        help="Map source target offsets directly or by source/generated FPS.",
    )
    args = parser.parse_args()

    cv2, np = import_cv2_np()
    manifest = read_manifest(args.manifest)
    if args.offset:
        manifest = manifest[args.offset :]
    if args.limit is not None:
        manifest = manifest[: args.limit]
    reports = [(label, read_report(path)) for label, path in parse_reports(args.reports)]

    section("EVALUATE GENERATED VIDEO REPORTS")
    kv("manifest", args.manifest)
    kv("rows", len(manifest))
    kv("reports", ", ".join(label for label, _ in reports))
    kv("repeat_last", args.include_repeat_last)
    kv("offset_mode", args.offset_mode)

    results = []
    for row in manifest:
        sample_id = str(row.get("sample_id", ""))
        if args.include_repeat_last:
            results.append(evaluate_candidate(cv2, np, row, "repeat_last", None, args.offset_mode))
        for label, video_map in reports:
            results.append(
                evaluate_candidate(cv2, np, row, label, video_map.get(sample_id), args.offset_mode)
            )

    write_csv(args.out_csv, results)
    section("SUMMARY")
    print_summary(np, results)
    section("OUTPUT")
    kv("csv", args.out_csv)


if __name__ == "__main__":
    main()

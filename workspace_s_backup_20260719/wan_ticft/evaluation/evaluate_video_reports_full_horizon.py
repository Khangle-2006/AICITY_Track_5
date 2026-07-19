"""Evaluate generated reports on every future frame up to the last target.

The regular validation scorer compares only the sparse target frames stored in
the manifest. That is useful for quick gates, but it can overrate generators
whose intermediate frames flicker or drift. This scorer decodes the source
video itself and scores the full continuous horizon:

    last_condition + 1 ... max(target_indices)

Candidate report videos are expected to include the guard/last-condition frame
at index 0, so source offset +k is compared with generated video frame k.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from baseline.frame_io import import_cv2_np, read_rgb
from baseline.metrics import mean_absdiff, psnr, ssim_gray
from wan_ticft.evaluation.evaluate_video_reports import read_report


FIELDS = [
    "sample_id",
    "candidate",
    "video",
    "horizon_frames",
    "available_frames",
    "psnr",
    "ssim",
    "absdiff",
    "status",
    "notes",
]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def parse_reports(values: list[str]) -> list[tuple[str, Path]]:
    reports = []
    for value in values:
        if "=" in value:
            label, raw = value.split("=", 1)
        else:
            raw = value
            label = Path(value).parent.name
        reports.append((label.strip(), Path(raw).expanduser()))
    return reports


def rotation_candidates(cv2, frame):
    return {
        "none": frame,
        "clockwise": cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE),
        "counterclockwise": cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE),
    }


def choose_orientation(cv2, np, frame_bgr, reference_rgb):
    if reference_rgb is None:
        return "none", 0.0
    scored = []
    for name, candidate in rotation_candidates(cv2, frame_bgr).items():
        rgb = cv2.cvtColor(candidate, cv2.COLOR_BGR2RGB)
        if rgb.shape[:2] != reference_rgb.shape[:2]:
            rgb = cv2.resize(rgb, (reference_rgb.shape[1], reference_rgb.shape[0]))
        error = float(np.abs(rgb.astype("float32") - reference_rgb.astype("float32")).mean())
        scored.append((error, name))
    return min(scored, key=lambda item: item[0])


def apply_orientation(cv2, frame_bgr, orientation: str):
    if orientation == "clockwise":
        return cv2.rotate(frame_bgr, cv2.ROTATE_90_CLOCKWISE)
    if orientation == "counterclockwise":
        return cv2.rotate(frame_bgr, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return frame_bgr


def decode_source_future(cv2, np, row: dict[str, Any]) -> tuple[list[Any], str, float]:
    indices = row.get("frame_indices") or {}
    condition = [int(value) for value in indices.get("condition") or []]
    target = [int(value) for value in indices.get("target") or []]
    if not condition or not target:
        return [], "none", 0.0
    last_condition = int(condition[-1])
    last_target = max(target)
    reference_paths = (row.get("frames") or {}).get("condition") or []
    reference = read_rgb(cv2, Path(reference_paths[-1])) if reference_paths else None

    cap = cv2.VideoCapture(str(row["video_path"]))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open source video {row['video_path']}")
    cap.set(cv2.CAP_PROP_POS_FRAMES, last_condition)
    ok, last_bgr = cap.read()
    if not ok or last_bgr is None:
        cap.release()
        raise RuntimeError(f"cannot read source frame {last_condition} from {row['video_path']}")
    orientation_error, orientation = choose_orientation(cv2, np, last_bgr, reference)

    frames = []
    for frame_index in range(last_condition + 1, last_target + 1):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, bgr = cap.read()
        if not ok or bgr is None:
            break
        bgr = apply_orientation(cv2, bgr, orientation)
        frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    cap.release()
    return frames, orientation, orientation_error


def decode_video_rgb(cv2, path: Path, count: int) -> list[Any]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return []
    frames = []
    for index in range(count + 1):
        ok, bgr = cap.read()
        if not ok or bgr is None:
            break
        if index > 0:
            frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    cap.release()
    return frames


def resize_like(cv2, pred, target):
    th, tw = target.shape[:2]
    if pred.shape[:2] == (th, tw):
        return pred
    return cv2.resize(pred, (tw, th), interpolation=cv2.INTER_CUBIC)


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


def evaluate_one(cv2, np, row: dict[str, Any], label: str, video: Path | None) -> dict[str, Any]:
    targets, orientation, orientation_error = decode_source_future(cv2, np, row)
    sample_id = str(row.get("sample_id", ""))
    if not targets:
        return {
            "sample_id": sample_id,
            "candidate": label,
            "video": str(video or ""),
            "horizon_frames": 0,
            "available_frames": 0,
            "status": "bad",
            "notes": "no source future frames",
        }
    if label == "repeat_last":
        reference_paths = (row.get("frames") or {}).get("condition") or []
        last = read_rgb(cv2, Path(reference_paths[-1])) if reference_paths else None
        preds = [last] * len(targets) if last is not None else []
    elif video is None:
        preds = []
    else:
        preds = decode_video_rgb(cv2, video, len(targets))

    status = "ok" if len(preds) >= len(targets) else "bad"
    if status == "ok":
        preds = preds[: len(targets)]
        p, s, d = score_frames(cv2, np, preds, targets)
    else:
        p, s, d = 0.0, 0.0, 0.0
    return {
        "sample_id": sample_id,
        "candidate": label,
        "video": str(video or ""),
        "horizon_frames": len(targets),
        "available_frames": len(preds),
        "psnr": f"{p:.6f}" if status == "ok" else "",
        "ssim": f"{s:.6f}" if status == "ok" else "",
        "absdiff": f"{d:.6f}" if status == "ok" else "",
        "status": status,
        "notes": f"source_orientation={orientation}; condition_absdiff={orientation_error:.3f}",
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows({field: row.get(field, "") for field in FIELDS} for row in rows)


def print_summary(np, rows: list[dict[str, Any]]) -> None:
    print(f"{'Candidate':<28} {'N':>5} {'PSNR':>10} {'SSIM':>10} {'AbsDiff':>10}")
    print("-" * 68)
    for label in sorted({row["candidate"] for row in rows}):
        items = [row for row in rows if row["candidate"] == label and row["status"] == "ok"]
        if not items:
            continue
        print(
            f"{label:<28} {len(items):>5} "
            f"{np.mean([float(row['psnr']) for row in items]):>10.4f} "
            f"{np.mean([float(row['ssim']) for row in items]):>10.4f} "
            f"{np.mean([float(row['absdiff']) for row in items]):>10.4f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--reports", nargs="+", required=True, help="Use label=/path/to/report.csv.")
    parser.add_argument("--out_csv", type=Path, required=True)
    parser.add_argument("--include_repeat_last", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    cv2, np = import_cv2_np()
    manifest = read_jsonl(args.manifest)
    if args.limit:
        manifest = manifest[: args.limit]
    reports = [(label, read_report(path)) for label, path in parse_reports(args.reports)]

    rows = []
    for row in manifest:
        sample_id = str(row.get("sample_id", ""))
        if args.include_repeat_last:
            rows.append(evaluate_one(cv2, np, row, "repeat_last", None))
        for label, report in reports:
            rows.append(evaluate_one(cv2, np, row, label, report.get(sample_id)))

    write_csv(args.out_csv, rows)
    print_summary(np, rows)
    print(f"output={args.out_csv}")


if __name__ == "__main__":
    main()

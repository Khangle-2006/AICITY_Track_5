from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path
from typing import Any

from baseline.frame_io import import_cv2_np, read_rgb
from wan_ticft.evaluation.evaluate_bdd_ego_motion import region_predictions
from wan_ticft.evaluation.evaluate_cnn_ltx_region_selector import target_offsets


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_images(cv2, paths: list[str]):
    existing = [Path(path) for path in paths if Path(path).exists()]
    return [image for path in existing if (image := read_rgb(cv2, path)) is not None]


def load_video_indices(cv2, row: dict[str, Any], indices: list[int]):
    video_path = Path(str(row.get("video_path") or ""))
    if not video_path.exists() or not indices:
        return []
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return []
    frames = []
    for index in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(index))
        ok, frame = cap.read()
        if not ok or frame is None:
            cap.release()
            return []
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    return frames


def load_condition_target_frames(cv2, row: dict[str, Any]):
    paths = row.get("frames") or {}
    conditions = load_images(cv2, paths.get("condition") or [])
    targets = load_images(cv2, paths.get("target") or [])
    indices = row.get("frame_indices") or {}
    if len(conditions) < 2:
        conditions = load_video_indices(cv2, row, indices.get("condition") or [])
    if not targets:
        targets = load_video_indices(cv2, row, indices.get("target") or [])
    return conditions, targets


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
    parser = argparse.ArgumentParser(description="Materialize cheap region-motion base videos for BDD train manifests.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--estimate_width", type=int, default=640)
    parser.add_argument("--clean", action="store_true")
    args = parser.parse_args()

    cv2, np = import_cv2_np()
    if args.clean and args.out_dir.exists():
        shutil.rmtree(args.out_dir)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    videos_dir = args.out_dir / "videos"

    rows = [row for row in read_jsonl(args.manifest) if row.get("view_type") == "bdd" and row.get("status", "ok") == "ok"]
    if args.limit:
        rows = rows[: args.limit]
    print(f"selected_bdd_rows={len(rows)} from {args.manifest}", flush=True)
    report = []
    selected_rows = []
    existing_report = args.out_dir / "report.csv"
    if existing_report.exists() and not args.clean:
        with existing_report.open("r", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                video = Path(row.get("video", ""))
                if row.get("sample_id") and video.exists():
                    report.append(row)
        if report:
            print(f"resume_existing={len(report)}", flush=True)
    done = {row["sample_id"] for row in report}
    for position, row in enumerate(rows, start=1):
        sample_id = row["sample_id"]
        if sample_id in done:
            selected_rows.append(row)
            print(f"[{position}/{len(rows)}] {sample_id} | resume existing", flush=True)
            continue
        conditions, targets = load_condition_target_frames(cv2, row)
        if len(conditions) < 2 or not targets:
            print(
                f"[skip] {sample_id}: conditions={len(conditions)} targets={len(targets)}",
                flush=True,
            )
            continue
        indices = row.get("frame_indices") or {}
        condition_indices = indices.get("condition") or []
        history_gap = max(1, int(condition_indices[-1]) - int(condition_indices[-2])) if len(condition_indices) >= 2 else 1
        offsets = target_offsets(row, len(targets))
        frames = region_predictions(
            cv2,
            np,
            conditions[-2],
            conditions[-1],
            offsets,
            history_gap,
            args.estimate_width,
        )
        path = videos_dir / f"{sample_id}.mp4"
        fps = float((row.get("frame_indices") or {}).get("fps") or 30.0)
        write_video(cv2, frames, path, fps)
        report.append(
            {
                "sample_id": sample_id,
                "video": str(path),
                "status": "ok",
                "future_frames": len(frames),
            }
        )
        selected_rows.append(row)
        done.add(sample_id)
        print(f"[{position}/{len(rows)}] {sample_id} | ok", flush=True)

    if not report:
        raise RuntimeError("no region-motion videos written")
    with (args.out_dir / "report.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["sample_id", "video", "status", "future_frames"])
        writer.writeheader()
        writer.writerows(report)
    with (args.out_dir / "selected_manifest.jsonl").open("w", encoding="utf-8") as handle:
        for row in selected_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"report={args.out_dir / 'report.csv'}")


if __name__ == "__main__":
    main()

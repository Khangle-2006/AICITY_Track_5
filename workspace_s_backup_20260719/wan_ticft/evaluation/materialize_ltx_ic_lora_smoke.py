"""Materialize paired BDD history-reference and future-target videos for IC-LoRA."""

from __future__ import annotations

import argparse
import json
import shutil
from collections import defaultdict
from pathlib import Path

from baseline.frame_io import import_cv2_np


PHASE_FRACTIONS = {"0": 0.0, "1": 0.25, "2": 0.5, "3": 0.75, "4": 1.0}


def read_rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def safe_name(text: str) -> str:
    return "".join(char if char.isalnum() or char in "._-" else "_" for char in text)[:170]


def write_video(cv2, frames, path: Path, fps: float, width: int, height: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"cannot create {path}")
    for frame in frames:
        writer.write(cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA))
    writer.release()


def spread_history(history, count: int):
    return [history[min(len(history) - 1, index * len(history) // count)] for index in range(count)]


def read_clip(cv2, video_path: Path, start: int, count: int):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open {video_path}")
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    frames = []
    for index in range(start, start + count):
        ok, frame = cap.read()
        if not ok or frame is None:
            cap.release()
            raise RuntimeError(f"cannot read {video_path} at frame {index}")
        frames.append(frame)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
    cap.release()
    return frames, fps


def candidate_starts(total: int, frames: int, history_frames: int, fraction: float) -> list[int]:
    minimum = history_frames
    maximum = total - frames
    if maximum < minimum:
        return []
    preferred = round(minimum + (maximum - minimum) * fraction)
    starts = [preferred]
    for amount in (frames // 4, frames // 2, frames, frames * 2):
        starts.extend([max(minimum, preferred - amount), min(maximum, preferred + amount)])
    starts.extend([minimum, (minimum + maximum) // 2, maximum])
    return list(dict.fromkeys(starts))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--per_phase", type=int, default=4)
    parser.add_argument("--frames", type=int, default=121)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=288)
    parser.add_argument("--fps", type=float, default=25.0)
    parser.add_argument("--history_frames", type=int, default=4)
    parser.add_argument("--clean", action="store_true")
    args = parser.parse_args()

    if args.frames % 8 != 1:
        raise SystemExit("frames must satisfy 8n+1")
    if args.clean and args.out_dir.exists():
        shutil.rmtree(args.out_dir)

    cv2, _ = import_cv2_np()
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in read_rows(args.manifest):
        if row.get("view_type") == "bdd":
            grouped[str(row.get("phase", ""))].append(row)

    metadata, report, skipped = [], [], []
    for phase in ("0", "1", "2", "3", "4"):
        written = 0
        for row in grouped[phase]:
            if written >= args.per_phase:
                break
            video_path = Path(row["video_path"])
            cap = cv2.VideoCapture(str(video_path))
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            cap.release()

            history = target = None
            fps = args.fps
            target_start = 0
            errors = []
            for proposed_start in candidate_starts(
                total, args.frames, args.history_frames, PHASE_FRACTIONS[phase]
            ):
                try:
                    history, fps = read_clip(
                        cv2, video_path, proposed_start - args.history_frames, args.history_frames
                    )
                    target, fps = read_clip(cv2, video_path, proposed_start, args.frames)
                    target_start = proposed_start
                    break
                except RuntimeError as exc:
                    errors.append(str(exc))
            if history is None or target is None:
                skipped.append(
                    {
                        "sample_id": row["sample_id"],
                        "phase": phase,
                        "source_video": str(video_path),
                        "reported_frames": total,
                        "reason": errors[-1] if errors else "source video is too short",
                    }
                )
                print(f"skip phase={phase} {row['sample_id']}: {skipped[-1]['reason']}", flush=True)
                continue

            name = f"{len(report):03d}_{safe_name(row['sample_id'])}.mp4"
            reference_path = args.out_dir / "references" / name
            target_path = args.out_dir / "targets" / name
            reference = spread_history(history, args.frames)
            write_video(cv2, reference, reference_path, fps, args.width, args.height)
            write_video(cv2, target, target_path, fps, args.width, args.height)

            caption = (
                "Continue the observed forward-facing traffic video into the near future. "
                "Preserve camera motion, road geometry, object identity, and realistic traffic motion."
            )
            metadata.append(
                {
                    "caption": caption,
                    "media_path": f"targets/{name}",
                    "reference_path": f"references/{name}",
                }
            )
            report.append(
                {
                    "sample_id": row["sample_id"],
                    "phase": phase,
                    "source_video": str(video_path),
                    "source_total_frames": total,
                    "history_start": target_start - args.history_frames,
                    "target_start": target_start,
                    "history_frames": len(history),
                    "target_frames": len(target),
                    "reference": str(reference_path),
                    "target": str(target_path),
                }
            )
            written += 1
            print(f"[{len(report)}/{args.per_phase * 5}] phase={phase} {row['sample_id']}", flush=True)
        if written != args.per_phase:
            raise RuntimeError(f"phase {phase}: materialized {written}/{args.per_phase}")

    dataset = args.out_dir / "dataset.jsonl"
    dataset.parent.mkdir(parents=True, exist_ok=True)
    dataset.write_text("".join(json.dumps(row) + "\n" for row in metadata), encoding="utf-8")
    validation_reference = args.out_dir / "validation_reference.mp4"
    shutil.copy2(Path(report[0]["reference"]), validation_reference)
    (args.out_dir / "report.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in report), encoding="utf-8"
    )
    (args.out_dir / "skipped.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in skipped), encoding="utf-8"
    )
    (args.out_dir / "summary.json").write_text(
        json.dumps(
            {
                "rows": len(report),
                "frames": args.frames,
                "history_frames": args.history_frames,
                "skipped": len(skipped),
                "dataset": str(dataset),
                "validation_reference": str(validation_reference),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(json.dumps({"rows": len(report), "status": "ok"}, indent=2))


if __name__ == "__main__":
    main()

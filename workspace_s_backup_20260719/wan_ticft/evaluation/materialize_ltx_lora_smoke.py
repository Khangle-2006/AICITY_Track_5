from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from baseline.frame_io import import_cv2_np


PHASE_FRACTIONS = {"0": 0.0, "1": 0.25, "2": 0.5, "3": 0.75, "4": 1.0}


def safe_name(text: str) -> str:
    return "".join(char if char.isalnum() or char in "._-" else "_" for char in text)[:180]


def candidate_rows(manifest: Path) -> dict[str, list[dict]]:
    groups: dict[str, list[dict]] = defaultdict(list)
    with manifest.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            phase = str(row.get("phase", ""))
            if row.get("view_type") == "bdd" and phase in PHASE_FRACTIONS:
                groups[phase].append(row)
    return groups


def read_clip(cv2, video_path: Path, start: int, frames: int):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open {video_path}")
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    result = []
    for index in range(start, start + frames):
        ok, frame = cap.read()
        if not ok or frame is None:
            cap.release()
            raise RuntimeError(f"cannot read {video_path} at frame {index}")
        result.append(frame)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    cap.release()
    return result, fps


def candidate_starts(total: int, frames: int, fraction: float) -> list[int]:
    maximum = max(0, total - frames)
    preferred = round(maximum * fraction)
    starts = [preferred]
    # Some MP4s report more frames than their final decodable timestamp. Move
    # progressively earlier before rejecting the source video.
    for amount in (frames // 4, frames // 2, frames, frames * 2, frames * 3):
        starts.append(max(0, preferred - amount))
    starts.extend([maximum // 2, 0])
    return list(dict.fromkeys(starts))


def write_video(cv2, frames, path: Path, fps: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    height, width = frames[0].shape[:2]
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"cannot create {path}")
    for frame in frames:
        writer.write(frame)
    writer.release()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--per_phase", type=int, default=4)
    parser.add_argument("--frames", type=int, default=121)
    parser.add_argument("--clean", action="store_true")
    args = parser.parse_args()

    cv2, _ = import_cv2_np()
    if args.clean and args.out_dir.exists():
        import shutil

        shutil.rmtree(args.out_dir)
    videos_dir = args.out_dir / "videos"
    videos_dir.mkdir(parents=True, exist_ok=True)

    metadata = []
    report = []
    skipped = []
    groups = candidate_rows(args.manifest)
    for phase, fraction in PHASE_FRACTIONS.items():
        phase_ok = 0
        for row in groups[phase]:
            if phase_ok >= args.per_phase:
                break
            video_path = Path(row["video_path"])
            cap = cv2.VideoCapture(str(video_path))
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            cap.release()
            if total < args.frames:
                skipped.append({"sample_id": row["sample_id"], "reason": f"only {total} reported frames"})
                continue

            frames = None
            fps = 0.0
            start = 0
            errors = []
            for proposed_start in candidate_starts(total, args.frames, fraction):
                try:
                    frames, fps = read_clip(cv2, video_path, proposed_start, args.frames)
                    start = proposed_start
                    break
                except RuntimeError as exc:
                    errors.append(str(exc))
            if frames is None:
                skipped.append({"sample_id": row["sample_id"], "reason": errors[-1]})
                print(f"skip phase={phase} {row['sample_id']}: {errors[-1]}")
                continue

            number = len(report)
            filename = f"{number:03d}_{safe_name(row['sample_id'])}.mp4"
            output = videos_dir / filename
            write_video(cv2, frames, output, fps)

            caption = (
                "Continue the observed forward-facing driving camera video. "
                "Preserve the road layout, lighting, camera viewpoint, vehicle identities, and pedestrian identities. "
                "Generate realistic ego motion and temporally consistent near-future traffic motion."
            )
            metadata.append({"caption": caption, "media_path": f"videos/{filename}"})
            report.append(
                {
                    "sample_id": row["sample_id"],
                    "phase": phase,
                    "source_video": str(video_path),
                    "source_total_frames": total,
                    "source_start": start,
                    "clip_frames": len(frames),
                    "fps": fps,
                    "output": str(output),
                }
            )
            phase_ok += 1
            print(f"[{len(report)}/{args.per_phase * 5}] phase={phase} start={start} {row['sample_id']}")
        if phase_ok != args.per_phase:
            raise RuntimeError(f"phase {phase}: materialized only {phase_ok}/{args.per_phase} valid clips")

    with (args.out_dir / "dataset.jsonl").open("w", encoding="utf-8") as handle:
        for row in metadata:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    with (args.out_dir / "report.jsonl").open("w", encoding="utf-8") as handle:
        for row in report:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    with (args.out_dir / "skipped.jsonl").open("w", encoding="utf-8") as handle:
        for row in skipped:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    (args.out_dir / "summary.json").write_text(
        json.dumps(
            {
                "rows": len(report),
                "frames_per_clip": args.frames,
                "phases": {phase: args.per_phase for phase in PHASE_FRACTIONS},
                "skipped": len(skipped),
                "dataset": str(args.out_dir / "dataset.jsonl"),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"dataset: {args.out_dir / 'dataset.jsonl'}")


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
import subprocess
from pathlib import Path

from baseline.frame_io import import_cv2_np, numeric_pngs
from baseline.verify import verify_sample_output


REPORT_FIELDS = [
    "sample_id",
    "output_name",
    "view_type",
    "phase",
    "history_frames",
    "target_N",
    "generated_total_frames",
    "output_frames",
    "status",
    "notes",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate BDD test frames with LTX-Video.")
    parser.add_argument("--test_csv", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--ltx_repo", type=Path, required=True)
    parser.add_argument("--ltx_python", type=Path, required=True)
    parser.add_argument("--pipeline_config", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--max_history", type=int, default=41)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument(
        "--prompt_mode",
        choices=["legacy", "minimal", "traffic_detail", "geometry_lock", "copy_bias", "dynamic_geometry"],
        default="legacy",
    )
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def read_rows(path: Path, offset: int, limit: int) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as f:
        rows = [row for row in csv.DictReader(f) if row.get("view_type") == "bdd"]
    return rows[offset : offset + limit if limit else None]


def compatible_history_length(frame_count: int, maximum: int) -> int:
    frame_count = min(frame_count, maximum)
    return ((frame_count - 1) // 8) * 8 + 1 if frame_count >= 9 else 0


def compatible_total_length(minimum: int) -> int:
    return int(math.ceil((minimum - 1) / 8) * 8 + 1)


def legacy_prompt(row: dict[str, str]) -> str:
    pedestrian = row.get("caption_pedestrian", "").split()[:45]
    vehicle = row.get("caption_vehicle", "").split()[:45]
    return " ".join(
        [
            "Continue the observed forward-facing traffic video.",
            "Preserve the same road, weather, lighting, camera viewpoint, and object identities.",
            "Pedestrian:",
            *pedestrian,
            "Vehicle:",
            *vehicle,
            "Maintain sharp geometry, realistic forward motion, and temporal consistency.",
        ]
    )


def minimal_prompt() -> str:
    return " ".join(
        [
            "Continue the observed video from the same forward-facing driving camera.",
            "Keep the same road, lighting, camera viewpoint, and visible objects.",
            "Preserve the observed motion.",
            "Do not add new people or vehicles.",
        ]
    )


def traffic_detail_prompt() -> str:
    return " ".join(
        [
            "The video is continuous realistic dashcam footage from a moving vehicle on an urban road.",
            "Continue the observed scene for the next few seconds with the same camera viewpoint, same focal length, same exposure, same weather, and same lighting.",
            "The ego vehicle moves smoothly forward with natural low-amplitude dashcam vibration; the road perspective expands consistently from the exact last observed frame.",
            "Preserve the lane geometry, curb lines, sidewalks, buildings, trees, traffic lights, parked vehicles, moving vehicles, pedestrians, shadows, reflections, and background layout.",
            "Vehicles ahead keep realistic speed and lane position, with no sudden jumps, no new vehicles, and no disappearing objects.",
            "Pedestrians, if visible, continue their current motion naturally.",
            "Keep the whole frame sharp and documentary-realistic: crisp lane markings, detailed asphalt texture, readable object boundaries, stable road edges, clear vehicle contours, natural motion blur only from real movement, no dreamy softness, no cinematic bokeh, no stylization.",
            "The shot is a single unbroken traffic-camera continuation with physically plausible motion and temporally stable details.",
        ]
    )


def geometry_lock_prompt() -> str:
    return " ".join(
        [
            "This is strict video continuation, not new scene generation.",
            "Start exactly from the final observed frame and preserve the exact camera calibration, focal length, horizon line, road perspective, lane markings, curb lines, sidewalks, building facades, trees, poles, parked cars, shadows, object identities, object scale, and object positions.",
            "Do not redesign, recompose, restyle, stretch, rotate, zoom, brighten, replace, shift, or reinterpret the scene.",
            "Static background objects may move only by the tiny parallax implied by the observed ego-vehicle motion.",
            "Already-visible vehicles and pedestrians continue with subtle physically plausible motion.",
            "No new objects appear and no existing objects disappear.",
            "Prioritize exact geometric consistency and temporal alignment over visual creativity.",
        ]
    )


def copy_bias_prompt() -> str:
    return " ".join(
        [
            "Continue the exact same dashcam video with minimal change from the final observed frame.",
            "The near future is short: keep the scene nearly identical, with only small smooth ego-motion and small actor motion.",
            "Copy the road geometry, lane markings, building edges, sidewalks, trees, parked vehicles, pedestrians, shadows, colors, exposure, and camera viewpoint from the final observed frame.",
            "Do not invent new details, do not move static objects, do not alter perspective, and do not change object identities.",
            "Keep the video realistic, sharp, stable, and temporally smooth.",
        ]
    )


def dynamic_geometry_prompt(row: dict[str, str]) -> str:
    phase = str(row.get("phase", ""))
    context = " ".join(
        [
            "Pedestrian:",
            " ".join(row.get("caption_pedestrian", "").split()[:50]),
            "Vehicle:",
            " ".join(row.get("caption_vehicle", "").split()[:60]),
        ]
    )
    if phase in {"0", "1"}:
        motion = "The future should change only slightly from the final observed frame; use conservative low-amplitude motion."
    elif phase == "2":
        motion = "Continue the observed motion smoothly while keeping object positions and road perspective tightly aligned."
    else:
        motion = "Continue later-phase motion smoothly, but do not introduce large camera or layout changes."
    return " ".join(
        [
            "Strictly continue this exact traffic video from the final observed frame.",
            motion,
            "Use the sample description only to constrain existing actors, not to create a new scene.",
            context,
            "Preserve exact road geometry, lane markings, curb lines, sidewalks, building facades, trees, poles, parked cars, traffic lights, shadows, object identities, object scale, camera calibration, focal length, and horizon line.",
            "Do not recompose, redesign, replace, shift, zoom, rotate, restyle, add objects, or remove objects.",
            "Prioritize geometric alignment with the final frame and target-like timing over cinematic quality.",
        ]
    )


def short_prompt(row: dict[str, str], prompt_mode: str) -> str:
    if prompt_mode == "minimal":
        return minimal_prompt()
    if prompt_mode == "traffic_detail":
        return traffic_detail_prompt()
    if prompt_mode == "geometry_lock":
        return geometry_lock_prompt()
    if prompt_mode == "copy_bias":
        return copy_bias_prompt()
    if prompt_mode == "dynamic_geometry":
        return dynamic_geometry_prompt(row)
    return legacy_prompt(row)


def write_video(cv2, frame_paths: list[Path], path: Path, fps: float) -> None:
    first = cv2.imread(str(frame_paths[0]), cv2.IMREAD_COLOR)
    if first is None:
        raise RuntimeError(f"cannot read {frame_paths[0]}")
    height, width = first.shape[:2]
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"cannot create {path}")
    for frame_path in frame_paths:
        frame = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
        if frame is None:
            writer.release()
            raise RuntimeError(f"cannot read {frame_path}")
        if frame.shape[:2] != (height, width):
            frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
        writer.write(frame)
    writer.release()


def write_report(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".tmp")
    with temp.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=REPORT_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    temp.replace(path)


def extract_future_frames(cv2, video_path: Path, history_count: int, target_n: int, out_dir: Path) -> int:
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open generated video {video_path}")
    generated = 0
    future_index = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if generated >= history_count - 1 and future_index < target_n:
            frame = cv2.resize(frame, (1280, 720), interpolation=cv2.INTER_LANCZOS4)
            if not cv2.imwrite(str(out_dir / f"{future_index}.png"), frame):
                cap.release()
                raise RuntimeError(f"cannot write output frame {future_index}")
            future_index += 1
        generated += 1
    cap.release()
    if future_index != target_n:
        raise RuntimeError(f"generated only {future_index}/{target_n} future frames")
    return generated


def main() -> None:
    args = parse_args()
    args.test_csv = args.test_csv.resolve()
    args.out_dir = args.out_dir.resolve()
    args.ltx_repo = args.ltx_repo.resolve()
    args.ltx_python = args.ltx_python.resolve()
    args.pipeline_config = args.pipeline_config.resolve()
    cv2, _ = import_cv2_np()
    rows = read_rows(args.test_csv, args.offset, args.limit)
    if not rows:
        raise RuntimeError("no BDD test rows selected")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    report_path = args.out_dir / "report.csv"
    report: list[dict[str, object]] = []

    for position, row in enumerate(rows, start=1):
        sample_id = row["sample_id"]
        output_name = row["source_folder"]
        target_n = int(row["target_N"])
        output_dir = args.out_dir / output_name
        existing = verify_sample_output(output_dir, target_n)
        if args.resume and existing.ok:
            result = {
                "sample_id": sample_id,
                "output_name": output_name,
                "view_type": "bdd",
                "phase": row.get("phase", ""),
                "history_frames": "",
                "target_N": target_n,
                "generated_total_frames": "",
                "output_frames": existing.frames_found,
                "status": "ok",
                "notes": "resume existing",
            }
            report.append(result)
            write_report(report_path, report)
            print(f"[{position}/{len(rows)}] {output_name} | resume existing", flush=True)
            continue

        history_paths = numeric_pngs(Path(row["input_dir"]))
        history_count = compatible_history_length(len(history_paths), args.max_history)
        if not history_count:
            raise RuntimeError(f"{sample_id}: fewer than 9 readable history PNGs")
        history_paths = history_paths[-history_count:]
        total_frames = compatible_total_length(history_count - 1 + target_n)

        history_video = args.out_dir / "_history" / f"{output_name}.mp4"
        write_video(cv2, history_paths, history_video, args.fps)
        raw_dir = args.out_dir / "_raw" / output_name
        if raw_dir.exists():
            shutil.rmtree(raw_dir)
        raw_dir.mkdir(parents=True)

        command = [
            str(args.ltx_python),
            str(args.ltx_repo / "inference.py"),
            "--prompt",
            short_prompt(row, args.prompt_mode),
            "--conditioning_media_paths",
            str(history_video),
            "--conditioning_start_frames",
            "0",
            "--conditioning_strengths",
            "1.0",
            "--height",
            "704",
            "--width",
            "1216",
            "--num_frames",
            str(total_frames),
            "--frame_rate",
            str(max(1, round(args.fps))),
            "--seed",
            str(args.seed),
            "--offload_to_cpu",
            "true",
            "--pipeline_config",
            str(args.pipeline_config),
            "--output_path",
            str(raw_dir),
        ]
        print(
            f"[{position}/{len(rows)}] {output_name} | history={history_count} "
            f"target={target_n} total={total_frames}",
            flush=True,
        )
        env = os.environ.copy()
        env["PYTHONPATH"] = str(args.ltx_repo)
        env["HF_HUB_DOWNLOAD_TIMEOUT"] = "3600"
        env["TOKENIZERS_PARALLELISM"] = "false"
        subprocess.run(command, check=True, env=env, cwd=args.ltx_repo)

        raw_videos = list(raw_dir.glob("*.mp4"))
        if len(raw_videos) != 1:
            raise RuntimeError(f"{sample_id}: expected one output video, found {len(raw_videos)}")
        generated = extract_future_frames(cv2, raw_videos[0], history_count, target_n, output_dir)
        verified = verify_sample_output(output_dir, target_n)
        if not verified.ok:
            raise RuntimeError(f"{sample_id}: output verification failed: {verified.notes}")

        result = {
            "sample_id": sample_id,
            "output_name": output_name,
            "view_type": "bdd",
            "phase": row.get("phase", ""),
            "history_frames": history_count,
            "target_N": target_n,
            "generated_total_frames": generated,
            "output_frames": verified.frames_found,
            "status": "ok",
            "notes": "",
        }
        report.append(result)
        write_report(report_path, report)

    with (args.out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump({"rows": len(report), "status": "ok", "report": str(report_path)}, f, indent=2)
    print(json.dumps({"rows": len(report), "status": "ok", "report": str(report_path)}, indent=2))


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
import subprocess
from pathlib import Path

from baseline.frame_io import import_cv2_np, read_rgb


def read_rows(
    path: Path,
    limit: int,
    phases: list[str] | None,
    view_type: str,
    sample_ids: list[str] | None,
) -> list[dict]:
    rows = []
    remaining = set(phases or [])
    wanted_samples = set(sample_ids or [])
    with path.open(encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            phase = str(row.get("phase", ""))
            sample_id = str(row.get("sample_id", ""))
            if (
                row.get("view_type") == view_type
                and (not remaining or phase in remaining)
                and (not wanted_samples or sample_id in wanted_samples)
            ):
                rows.append(row)
                remaining.discard(phase)
                if len(rows) == limit:
                    break
    return rows


def compatible_history_length(frame_count: int, maximum: int = 41) -> int:
    frame_count = min(frame_count, maximum)
    return ((frame_count - 1) // 8) * 8 + 1 if frame_count >= 9 else 0


def read_history(cv2, video_path: Path, last_index: int, count: int):
    start = last_index - count + 1
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open {video_path}")
    frames = []
    for index in range(start, last_index + 1):
        # Some BDD MP4s decode the wrong orientation/segment after a long
        # sequential seek. Match build_manifest.py by seeking every frame.
        cap.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, frame = cap.read()
        if not ok or frame is None:
            cap.release()
            raise RuntimeError(f"cannot read {video_path} at frame {index}")
        frames.append(frame)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    cap.release()
    return frames, fps


def read_history_sequential(cv2, video_path: Path, last_index: int, count: int):
    start = last_index - count + 1
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open {video_path}")
    frames = []
    source_index = 0
    while source_index <= last_index:
        ok, frame = cap.read()
        if not ok or frame is None:
            cap.release()
            raise RuntimeError(f"cannot sequentially read {video_path} at frame {source_index}")
        if source_index >= start:
            frames.append(frame)
        source_index += 1
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    cap.release()
    if len(frames) != count:
        raise RuntimeError(f"expected {count} sequential history frames, found {len(frames)}")
    return frames, fps


def write_video(cv2, frames, path: Path, fps: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    height, width = frames[0].shape[:2]
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"cannot create {path}")
    for frame in frames:
        writer.write(frame)
    writer.release()


def behavior_prompt(row: dict) -> str:
    prompt = str(row.get("prompt", ""))
    match = re.search(r"Vehicle state:\s*(.*?)(?:Predict the near-future frames|$)", prompt)
    vehicle = match.group(1) if match else ""
    motion = []
    for pattern in (
        r"(?:constant speed|speed)(?:\s+of|\s+is|\s+currently\s+at)?\s*\d+\s*km/h",
        r"(?:started|starts|starting|about)\s+to\s+turn\s+(?:left|right)",
        r"(?:turning|turns|turned)\s+(?:left|right)",
        r"(?:going|moving|travelling|traveling|proceeding|continues?)\s+straight(?:\s+ahead)?",
        r"(?:stationary|halted|not moving|accelerating|decelerating|about to stop)",
    ):
        found = re.search(pattern, vehicle, flags=re.IGNORECASE)
        if found:
            motion.append(found.group(0))
    motion_text = "; ".join(dict.fromkeys(item.lower() for item in motion))
    if not motion_text:
        motion_text = "continue the observed vehicle motion"
    return " ".join(
        [
            "Continue the observed video from the same forward-facing driving camera.",
            "Preserve the exact road, lighting, viewpoint, and identities of all visible objects.",
            f"Vehicle motion: {motion_text}.",
            "Continue only already visible actors with physically plausible motion.",
            "Do not add, remove, enlarge, or replace pedestrians or vehicles.",
            "Maintain sharp geometry and temporal consistency.",
        ]
    )


def minimal_prompt(row: dict) -> str:
    view_type = str(row.get("view_type", ""))
    camera = {
        "bdd": "forward-facing driving camera",
        "wts_vehicle_or_ip_camera": "traffic camera",
        "wts_overhead_or_fixed_camera": "fixed traffic camera",
    }.get(view_type, "traffic camera")
    return " ".join(
        [
            f"Continue the observed video from the same {camera}.",
            "Keep the same road, lighting, camera viewpoint, and visible objects.",
            "Preserve the observed motion.",
            "Do not add new people or vehicles.",
        ]
    )


def traffic_detail_prompt(row: dict) -> str:
    view_type = str(row.get("view_type", ""))
    if view_type == "wts_overhead_or_fixed_camera":
        return " ".join(
            [
                "The video is continuous realistic fixed traffic-surveillance footage from the same elevated camera.",
                "Continue the observed scene with a locked camera, fixed perspective, fixed background, stable exposure, and unchanged lighting.",
                "The road, crosswalk, lane markings, sidewalks, buildings, parked cars, shadows, and static background remain stable and sharp.",
                "Only already-visible pedestrians and vehicles continue moving along their current trajectories with realistic speed and smooth temporal motion.",
                "Do not introduce new people or cars.",
                "Do not change the camera angle, zoom, color tone, or background geometry.",
                "Preserve crisp object boundaries, clear road texture, stable contours, and documentary surveillance-video realism with no blur, no hallucinated objects, no stylized cinematic look, and no flicker.",
            ]
        )
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


def row_context(row: dict) -> str:
    prompt = str(row.get("prompt", "")).strip()
    if prompt:
        parts = []
        for label in ("Pedestrian state:", "Vehicle state:"):
            if label in prompt:
                tail = prompt.split(label, 1)[1]
                tail = tail.split("Vehicle state:", 1)[0] if label == "Pedestrian state:" else tail
                tail = tail.split("Predict the near-future frames", 1)[0]
                parts.append(" ".join(tail.split()[:70]))
        if parts:
            return " ".join(parts)
    pedestrian = " ".join(str(row.get("caption_pedestrian", "")).split()[:45])
    vehicle = " ".join(str(row.get("caption_vehicle", "")).split()[:55])
    return " ".join(part for part in [pedestrian, vehicle] if part)


def geometry_lock_prompt(row: dict) -> str:
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


def copy_bias_prompt(row: dict) -> str:
    return " ".join(
        [
            "Continue the exact same dashcam video with minimal change from the final observed frame.",
            "The near future is short: keep the scene nearly identical, with only small smooth ego-motion and small actor motion.",
            "Copy the road geometry, lane markings, building edges, sidewalks, trees, parked vehicles, pedestrians, shadows, colors, exposure, and camera viewpoint from the final observed frame.",
            "Do not invent new details, do not move static objects, do not alter perspective, and do not change object identities.",
            "Keep the video realistic, sharp, stable, and temporally smooth.",
        ]
    )


def dynamic_geometry_prompt(row: dict) -> str:
    phase = str(row.get("phase", ""))
    context = " ".join(row_context(row).split()[:34])
    if phase in {"0", "1"}:
        motion = "very small forward motion"
    elif phase in {"2"}:
        motion = "smooth observed motion"
    else:
        motion = "smooth later-phase motion"
    return " ".join(
        [
            "Exact dashcam continuation, not a new scene.",
            "Copy final-frame road geometry, camera viewpoint, focal length, horizon, lanes, curbs, buildings, trees, parked cars, traffic lights, shadows, colors, and object identities.",
            "No new objects, no disappearing objects, no restyling, no zoom, no camera rotation.",
            f"Motion: {motion}.",
            f"Actor context: {context}.",
            "Prioritize frame alignment over cinematic quality.",
        ]
    )


def dashcam_native_prompt(row: dict) -> str:
    return " ".join(
        [
            "Continue the same real dashcam recording from the final observed frame.",
            "Keep the existing lens, exposure, color response, windshield reflections, sensor noise, compression texture, motion blur, and low-detail distant background exactly consistent with the input video.",
            "Preserve the road layout, lane markings, curbs, buildings, parked cars, traffic lights, shadows, visible pedestrians, vehicle scale, and camera viewpoint without redesigning or beautifying the scene.",
            "The near future should look like ordinary unedited driving footage from the same camera, not a cinematic render or a polished generated image.",
            "Only continue the already observed ego motion and actor motion; do not introduce new objects, remove objects, sharpen surfaces artificially, repaint textures, change lighting, or clean up the image.",
        ]
    )


def short_prompt(row: dict, prompt_mode: str) -> str:
    if prompt_mode == "manifest":
        prompt = str(row.get("prompt", "")).strip()
        if prompt:
            return prompt
    if prompt_mode == "behavior":
        return behavior_prompt(row)
    if prompt_mode == "minimal":
        return minimal_prompt(row)
    if prompt_mode == "traffic_detail":
        return traffic_detail_prompt(row)
    if prompt_mode == "geometry_lock":
        return geometry_lock_prompt(row)
    if prompt_mode == "copy_bias":
        return copy_bias_prompt(row)
    if prompt_mode == "dynamic_geometry":
        return dynamic_geometry_prompt(row)
    if prompt_mode == "dashcam_native":
        return dashcam_native_prompt(row)
    pedestrian = str(row.get("caption_pedestrian", "")).split()[:45]
    vehicle = str(row.get("caption_vehicle", "")).split()[:45]
    view_type = str(row.get("view_type", ""))
    camera = {
        "bdd": "forward-facing driving camera",
        "wts_vehicle_or_ip_camera": "vehicle-mounted or roadside IP traffic camera",
        "wts_overhead_or_fixed_camera": "fixed overhead surveillance camera",
    }.get(view_type, "traffic camera")
    return " ".join(
        [
            f"Continue the observed video from the same {camera}.",
            "Preserve the same road, weather, lighting, camera viewpoint, and object identities.",
            "Pedestrian:",
            *pedestrian,
            "Vehicle:",
            *vehicle,
            "Maintain sharp geometry, realistic forward motion, and temporal consistency.",
        ]
    )


def align_history_to_reference(cv2, frames, reference):
    if reference is None:
        return frames, "unchecked", 0.0

    candidates = {
        "none": frames,
        "clockwise": [cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE) for frame in frames],
        "counterclockwise": [cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE) for frame in frames],
    }
    scored = []
    for name, candidate in candidates.items():
        last_rgb = cv2.cvtColor(candidate[-1], cv2.COLOR_BGR2RGB)
        if last_rgb.shape[:2] != reference.shape[:2]:
            last_rgb = cv2.resize(last_rgb, (reference.shape[1], reference.shape[0]))
        error = float(abs(last_rgb.astype("float32") - reference.astype("float32")).mean())
        scored.append((error, name, candidate))
    error, name, aligned = min(scored, key=lambda item: item[0])
    return aligned, name, error


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--ltx_repo", type=Path, required=True)
    parser.add_argument("--ltx_python", type=Path, required=True)
    parser.add_argument("--pipeline_config", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=2)
    parser.add_argument("--phases", nargs="*", default=None)
    parser.add_argument("--view_type", default="bdd")
    parser.add_argument("--sample_ids", nargs="*", default=None)
    parser.add_argument("--refine_video_dir", default=None)
    parser.add_argument("--refine_strength", type=float, default=0.0)
    parser.add_argument("--max_history_frames", type=int, default=41)
    parser.add_argument(
        "--prompt_mode",
        choices=[
            "legacy",
            "manifest",
            "behavior",
            "minimal",
            "traffic_detail",
            "geometry_lock",
            "copy_bias",
            "dynamic_geometry",
            "dashcam_native",
        ],
        default="legacy",
    )
    parser.add_argument("--future_anchor_offset", type=int, default=0)
    parser.add_argument("--future_anchor_strength", type=float, default=0.25)
    parser.add_argument("--height", type=int, default=704)
    parser.add_argument("--width", type=int, default=1216)
    parser.add_argument("--num_frames", type=int, default=121)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--skip_bad", action="store_true")
    args = parser.parse_args()

    cv2, _ = import_cv2_np()
    # Read a candidate pool and stop after enough successful outputs. This lets
    # --skip_bad backfill samples that cannot provide a valid history window.
    rows = read_rows(args.manifest, 0, args.phases, args.view_type, args.sample_ids)
    if len(rows) < args.limit:
        raise RuntimeError(f"found only {len(rows)} rows for view_type={args.view_type}")

    if args.out_dir.exists() and not args.resume:
        shutil.rmtree(args.out_dir)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    report = []
    completed_rows = []

    for position, row in enumerate(rows, start=1):
        if len(report) >= args.limit:
            break
        sample_id = row["sample_id"]
        condition = (row.get("frame_indices") or {}).get("condition") or []
        if not condition:
            raise RuntimeError(f"{sample_id}: missing condition indices")
        last_index = int(condition[-1])
        history_count = compatible_history_length(last_index + 1, args.max_history_frames)
        if not history_count:
            message = f"{sample_id}: fewer than 9 preceding frames"
            if args.skip_bad:
                print(f"[{position}/{len(rows)}] skip bad | {message}", flush=True)
                continue
            raise RuntimeError(message)

        source_fps = float((row.get("frame_indices") or {}).get("fps") or 30.0)
        future_path = args.out_dir / "videos" / f"{sample_id}.mp4"
        if args.resume and future_path.exists():
            print(f"[{position}/{len(rows)}] {sample_id} | resume existing", flush=True)
            report.append(
                {
                    "sample_id": sample_id,
                    "video": str(future_path),
                    "status": "ok",
                    "history_frames": history_count,
                    "source_fps": source_fps,
                    "future_frames": 81,
                }
            )
            completed_rows.append(row)
            continue

        video_path = Path(row["video_path"])
        decode_mode = "random_seek"
        try:
            frames, source_fps = read_history(cv2, video_path, last_index, history_count)
        except RuntimeError as exc:
            print(f"{sample_id}: random seek failed ({exc}); retrying sequential decode", flush=True)
            frames, source_fps = read_history_sequential(cv2, video_path, last_index, history_count)
            decode_mode = "sequential"
        condition_paths = (row.get("frames") or {}).get("condition") or []
        reference = read_rgb(cv2, Path(condition_paths[-1])) if condition_paths else None
        frames, orientation, mean_error = align_history_to_reference(cv2, frames, reference)
        if reference is not None and mean_error > 3.0 and decode_mode != "sequential":
            sequential_frames, sequential_fps = read_history_sequential(
                cv2, video_path, last_index, history_count
            )
            sequential_frames, sequential_orientation, sequential_error = align_history_to_reference(
                cv2, sequential_frames, reference
            )
            if sequential_error < mean_error:
                frames = sequential_frames
                source_fps = sequential_fps
                orientation = sequential_orientation
                mean_error = sequential_error
                decode_mode = "sequential"
        print(
            f"{sample_id}: selected history decode={decode_mode}, orientation={orientation}, "
            f"condition mean_absdiff={mean_error:.3f}",
            flush=True,
        )
        if reference is not None and mean_error > 3.0:
            message = (
                f"{sample_id}: extracted last history frame does not match manifest condition "
                f"(mean_absdiff={mean_error:.3f})"
            )
            if args.skip_bad:
                print(f"[{position}/{len(rows)}] skip bad | {message}", flush=True)
                continue
            raise RuntimeError(message)
        input_path = args.out_dir / "history_videos" / f"{sample_id}.mp4"
        write_video(cv2, frames, input_path, source_fps)
        anchor_path = args.out_dir / "anchor_images" / f"{sample_id}.png"
        if args.future_anchor_offset > 0:
            anchor_path.parent.mkdir(parents=True, exist_ok=True)
            if not cv2.imwrite(str(anchor_path), frames[-1]):
                raise RuntimeError(f"{sample_id}: cannot write future anchor image")

        raw_dir = args.out_dir / "raw" / sample_id
        if raw_dir.exists():
            shutil.rmtree(raw_dir)
        raw_dir.mkdir(parents=True)
        media_paths = [str(input_path)]
        start_frames = ["0"]
        strengths = ["1.0"]
        if args.future_anchor_offset > 0:
            media_paths.append(str(anchor_path))
            start_frames.append(str(history_count - 1 + args.future_anchor_offset))
            strengths.append(str(args.future_anchor_strength))
        if args.refine_video_dir and args.refine_strength > 0:
            _rp = Path(args.refine_video_dir) / f"{sample_id}.avi"
            if _rp.exists():
                media_paths.append(str(_rp))
                start_frames.append(str((history_count // 8) * 8))
                strengths.append(str(args.refine_strength))
            else:
                print(f"[refine] missing base video: {_rp}", flush=True)

        command = [
            str(args.ltx_python),
            str(args.ltx_repo / "inference.py"),
            "--prompt",
            short_prompt(row, args.prompt_mode),
            "--conditioning_media_paths",
            *media_paths,
            "--conditioning_start_frames",
            *start_frames,
            "--conditioning_strengths",
            *strengths,
            "--height",
            str(args.height),
            "--width",
            str(args.width),
            "--num_frames",
            str(args.num_frames),
            "--frame_rate",
            str(max(1, round(source_fps))),
            "--seed",
            "1",
            "--offload_to_cpu",
            "true",
            "--pipeline_config",
            str(args.pipeline_config),
            "--output_path",
            str(raw_dir),
        ]
        print(f"[{position}/{len(rows)}] {sample_id}", flush=True)
        env = os.environ.copy()
        env["PYTHONPATH"] = str(args.ltx_repo)
        env["HF_HUB_DOWNLOAD_TIMEOUT"] = "3600"
        env["TOKENIZERS_PARALLELISM"] = "false"
        subprocess.run(command, check=True, env=env)

        raw_videos = list(raw_dir.glob("*.mp4"))
        if len(raw_videos) != 1:
            raise RuntimeError(f"{sample_id}: expected one output video, found {len(raw_videos)}")
        cap = cv2.VideoCapture(str(raw_videos[0]))
        output_fps = float(cap.get(cv2.CAP_PROP_FPS) or source_fps)
        output_frames = []
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            output_frames.append(frame)
        cap.release()
        future_frames = output_frames[history_count - 1 :]
        write_video(cv2, future_frames, future_path, output_fps)
        report.append(
            {
                "sample_id": sample_id,
                "video": str(future_path),
                "status": "ok",
                "history_frames": history_count,
                "source_fps": source_fps,
                "future_frames": len(future_frames),
            }
        )
        completed_rows.append(row)

    if len(report) != args.limit:
        raise RuntimeError(f"completed only {len(report)}/{args.limit} requested rows")
    with (args.out_dir / "report.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=report[0].keys())
        writer.writeheader()
        writer.writerows(report)
    with (args.out_dir / "selected_manifest.jsonl").open("w", encoding="utf-8") as f:
        for row in completed_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps({"rows": len(report), "requested": args.limit, "candidates": len(rows), "status": "ok"}, indent=2))


if __name__ == "__main__":
    main()

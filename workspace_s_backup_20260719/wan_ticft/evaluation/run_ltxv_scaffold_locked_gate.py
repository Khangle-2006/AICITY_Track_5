from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

from baseline.frame_io import import_cv2_np, read_rgb
from wan_ticft.evaluation.evaluate_video_reports import read_report
from wan_ticft.evaluation.run_ltxv_val_gate import (
    align_history_to_reference,
    compatible_history_length,
    geometry_lock_prompt,
    read_history,
    read_history_sequential,
    read_rows,
    write_video,
)


NEGATIVE_PROMPT = (
    "new scene, camera angle change, hallucinated vehicle, duplicated pedestrian, "
    "object replacement, warped road, lane drift, double exposure, ghosting, smear, "
    "synthetic texture, dreamy blur, flicker, low quality"
)


def read_video_bgr(cv2, path: Path, count: int) -> tuple[list[Any], float]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open scaffold video: {path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    frames = []
    while len(frames) < count:
        ok, frame = cap.read()
        if not ok or frame is None:
            break
        frames.append(frame)
    cap.release()
    if not frames:
        raise RuntimeError(f"cannot read scaffold video: {path}")
    while len(frames) < count:
        frames.append(frames[-1].copy())
    return frames[:count], fps


def write_report(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["sample_id", "video", "status", "history_frames", "source_fps", "future_frames", "notes"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({field: row.get(field, "") for field in fields} for row in rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "LTX video-to-video/control gate: condition on real history plus a full future scaffold "
            "video, so the diffusion prior can denoise around existing geometry instead of generating from scratch."
        )
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--scaffold_report", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--ltx_repo", type=Path, default=Path("/workspace/third_party/LTX-Video"))
    parser.add_argument("--ltx_python", type=Path, default=Path("/workspace/experiments/wan_ticft/ltxv_env/bin/python"))
    parser.add_argument("--pipeline_config", type=Path, default=Path("/workspace/wan_ticft/evaluation/ltxv_2b_gate.yaml"))
    parser.add_argument("--limit", type=int, default=4)
    parser.add_argument("--sample_ids", nargs="*", default=None)
    parser.add_argument("--strengths", type=float, nargs="+", default=[0.85, 1.0])
    parser.add_argument("--future_condition_frames", type=int, default=33)
    parser.add_argument("--max_history_frames", type=int, default=41)
    parser.add_argument("--height", type=int, default=704)
    parser.add_argument("--width", type=int, default=1216)
    parser.add_argument("--seed", type=int, default=31)
    parser.add_argument("--image_cond_noise_scale", type=float, default=0.03)
    parser.add_argument(
        "--vae_tiling",
        choices=["0", "1"],
        default=None,
        help="Override LTX_VAE_TILING for the child LTX process. Defaults to the existing env or 1.",
    )
    parser.add_argument(
        "--use_input_media",
        action="store_true",
        help="Use the stitched history+scaffold clip as LTX video-to-video input_media_path.",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--skip_bad", action="store_true")
    args = parser.parse_args()

    if (args.future_condition_frames - 1) % 8 != 0:
        raise ValueError("--future_condition_frames must be 8n+1 for LTX conditioning")

    cv2, _np = import_cv2_np()
    rows = read_rows(args.manifest, 0, None, "bdd", args.sample_ids)
    scaffold_report = read_report(args.scaffold_report)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    labels = [f"scaf_s{int(round(strength * 100)):03d}" for strength in args.strengths]
    reports: dict[str, list[dict[str, Any]]] = {label: [] for label in labels}
    completed_rows: list[dict[str, Any]] = []

    for position, row in enumerate(rows, start=1):
        if len(completed_rows) >= args.limit:
            break
        sample_id = row["sample_id"]
        scaffold_video = scaffold_report.get(sample_id)
        if scaffold_video is None:
            message = f"{sample_id}: missing scaffold video in report"
            if args.skip_bad:
                print(f"[{position}/{len(rows)}] skip | {message}", flush=True)
                continue
            raise RuntimeError(message)

        condition = (row.get("frame_indices") or {}).get("condition") or []
        if not condition:
            raise RuntimeError(f"{sample_id}: missing condition indices")
        last_index = int(condition[-1])
        history_count = compatible_history_length(last_index + 1, args.max_history_frames)
        if history_count < 9:
            message = f"{sample_id}: fewer than 9 preceding frames"
            if args.skip_bad:
                print(f"[{position}/{len(rows)}] skip | {message}", flush=True)
                continue
            raise RuntimeError(message)

        source_fps = float((row.get("frame_indices") or {}).get("fps") or 30.0)
        try:
            history, source_fps = read_history(cv2, Path(row["video_path"]), last_index, history_count)
            decode_mode = "random_seek"
        except RuntimeError:
            history, source_fps = read_history_sequential(cv2, Path(row["video_path"]), last_index, history_count)
            decode_mode = "sequential"

        condition_paths = (row.get("frames") or {}).get("condition") or []
        reference = read_rgb(cv2, Path(condition_paths[-1])) if condition_paths else None
        history, orientation, mean_error = align_history_to_reference(cv2, history, reference)
        if reference is not None and mean_error > 3.0:
            message = f"{sample_id}: history/condition mismatch mean_absdiff={mean_error:.3f}"
            if args.skip_bad:
                print(f"[{position}/{len(rows)}] skip | {message}", flush=True)
                continue
            raise RuntimeError(message)

        work_dir = args.out_dir / "_inputs" / sample_id
        history_path = work_dir / "history.mp4"
        scaffold_path = work_dir / "scaffold_future.mp4"
        full_input_path = work_dir / "history_scaffold_input.mp4"
        if not (args.resume and history_path.exists() and scaffold_path.exists() and full_input_path.exists()):
            work_dir.mkdir(parents=True, exist_ok=True)
            write_video(cv2, history, history_path, source_fps)
            scaffold_frames, _ = read_video_bgr(cv2, scaffold_video, args.future_condition_frames)
            write_video(cv2, scaffold_frames, scaffold_path, source_fps)
            # Full V2V input has one shared anchor frame: history frames up to
            # the last observed frame, then scaffold future starting at that
            # same anchor position.
            write_video(cv2, [*history[:-1], *scaffold_frames], full_input_path, source_fps)

        # Start the scaffold at the last history latent frame. This is the same
        # alignment convention as run_ltxv_val_gate's future slicing.
        scaffold_start = history_count - 1
        num_frames = scaffold_start + args.future_condition_frames
        if (num_frames - 1) % 8 != 0:
            raise AssertionError(f"internal LTX frame count not 8n+1: {num_frames}")

        print(
            f"[{position}/{len(rows)}] {sample_id} | history={history_count} scaffold={args.future_condition_frames} "
            f"num_frames={num_frames} decode={decode_mode}/{orientation} diff={mean_error:.3f}",
            flush=True,
        )

        completed_rows.append(row)
        for label, strength in zip(labels, args.strengths):
            future_path = args.out_dir / label / "videos" / f"{sample_id}.mp4"
            if args.resume and future_path.exists():
                reports[label].append(
                    {
                        "sample_id": sample_id,
                        "video": str(future_path),
                        "status": "ok",
                        "history_frames": history_count,
                        "source_fps": f"{source_fps:.6f}",
                        "future_frames": "",
                        "notes": "resume",
                    }
                )
                print(f"  {label} resume", flush=True)
                continue

            raw_dir = args.out_dir / "_raw" / label / sample_id
            if raw_dir.exists():
                shutil.rmtree(raw_dir)
            raw_dir.mkdir(parents=True, exist_ok=True)
            command = [
                str(args.ltx_python),
                str(args.ltx_repo / "inference.py"),
                "--prompt",
                geometry_lock_prompt(row),
                "--negative_prompt",
                NEGATIVE_PROMPT,
                "--image_cond_noise_scale",
                f"{args.image_cond_noise_scale:.4f}",
                "--height",
                str(args.height),
                "--width",
                str(args.width),
                "--num_frames",
                str(num_frames),
                "--frame_rate",
                str(max(1, round(source_fps))),
                "--seed",
                str(args.seed + position),
                "--offload_to_cpu",
                "true",
                "--pipeline_config",
                str(args.pipeline_config),
                "--output_path",
                str(raw_dir),
            ]
            if args.use_input_media:
                command.extend(["--input_media_path", str(full_input_path)])
            else:
                command.extend(
                    [
                        "--conditioning_media_paths",
                        str(history_path),
                        str(scaffold_path),
                        "--conditioning_start_frames",
                        "0",
                        str(scaffold_start),
                        "--conditioning_strengths",
                        "1.0",
                        f"{strength:.4f}",
                    ]
                )
            env = os.environ.copy()
            env["PYTHONPATH"] = str(args.ltx_repo)
            env["HF_HUB_DOWNLOAD_TIMEOUT"] = "3600"
            env["TOKENIZERS_PARALLELISM"] = "false"
            env["LTX_VAE_TILING"] = args.vae_tiling if args.vae_tiling is not None else env.get("LTX_VAE_TILING", "1")
            subprocess.run(command, check=True, env=env)

            raw_videos = sorted(raw_dir.glob("*.mp4"))
            if len(raw_videos) != 1:
                raise RuntimeError(f"{sample_id}/{label}: expected one output video, found {len(raw_videos)}")
            cap = cv2.VideoCapture(str(raw_videos[0]))
            output_fps = float(cap.get(cv2.CAP_PROP_FPS) or source_fps)
            output_frames = []
            while True:
                ok, frame = cap.read()
                if not ok or frame is None:
                    break
                output_frames.append(frame)
            cap.release()
            future_frames = output_frames[scaffold_start:]
            write_video(cv2, future_frames, future_path, output_fps)
            reports[label].append(
                {
                    "sample_id": sample_id,
                    "video": str(future_path),
                    "status": "ok",
                    "history_frames": history_count,
                    "source_fps": f"{source_fps:.6f}",
                    "future_frames": len(future_frames),
                    "notes": f"scaffold={scaffold_video};strength={strength:.4f};noise={args.image_cond_noise_scale:.4f}",
                }
            )
            print(f"  {label} ok future_frames={len(future_frames)}", flush=True)

    if len(completed_rows) != args.limit:
        raise RuntimeError(f"completed only {len(completed_rows)}/{args.limit}")

    with (args.out_dir / "selected_manifest.jsonl").open("w", encoding="utf-8") as handle:
        for row in completed_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    for label, rows_for_label in reports.items():
        write_report(args.out_dir / label / "report.csv", rows_for_label)
    print(json.dumps({"status": "ok", "rows": len(completed_rows), "out_dir": str(args.out_dir)}, indent=2))


if __name__ == "__main__":
    main()

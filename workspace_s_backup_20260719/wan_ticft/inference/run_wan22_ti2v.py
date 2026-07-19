"""Run Wan2.2 TI2V inference from a materialized Track 5 manifest.

This wrapper intentionally calls the upstream Wan2.2 ``generate.py`` entrypoint.
It is slower than keeping the model resident, but it gives us a stable first
batch runner without depending on private internals of the Wan repository.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


def print_header(title: str) -> None:
    print("\n" + "=" * 100)
    print(title)
    print("=" * 100)


def print_kv(key: str, value: object, width: int = 42) -> None:
    print(f"{key:<{width}} {value}")


def slugify(text: str, max_len: int = 180) -> str:
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("_")
    return text[:max_len] or "sample"


def read_manifest(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid json: {exc}") from exc
            rows.append(row)
    return rows


def select_condition_image(row: dict[str, Any], policy: str) -> str:
    frames = row.get("frames") or {}
    condition = frames.get("condition") or []
    if not condition:
        raise ValueError("manifest row has no frames.condition entries")
    if policy == "first":
        return str(condition[0])
    if policy == "middle":
        return str(condition[len(condition) // 2])
    if policy == "last":
        return str(condition[-1])
    raise ValueError(f"unsupported image policy: {policy}")


def build_prompt(row: dict[str, Any], max_words: int | None) -> str:
    prompt = str(row.get("prompt") or "").strip()
    if not prompt:
        view = row.get("view_type", "traffic camera")
        prompt = f"{view} traffic scene, continue the motion naturally, realistic video"
    if max_words is not None and max_words > 0:
        words = prompt.split()
        if len(words) > max_words:
            prompt = " ".join(words[:max_words])
    return prompt


def run_one(
    *,
    row: dict[str, Any],
    idx: int,
    args: argparse.Namespace,
    out_video: Path,
    log_file: Path,
) -> dict[str, Any]:
    image = select_condition_image(row, args.image_policy)
    prompt = build_prompt(row, args.max_prompt_words)
    seed = args.base_seed + idx if args.base_seed >= 0 else -1

    cmd = [
        sys.executable,
        "generate.py",
        "--task",
        "ti2v-5B",
        "--size",
        args.size,
        "--ckpt_dir",
        str(args.ckpt_dir),
        "--image",
        image,
        "--prompt",
        prompt,
        "--frame_num",
        str(args.frame_num),
        "--sample_steps",
        str(args.sample_steps),
        "--sample_guide_scale",
        str(args.sample_guide_scale),
        "--sample_shift",
        str(args.sample_shift),
        "--base_seed",
        str(seed),
        "--save_file",
        str(out_video),
    ]
    if args.sample_solver:
        cmd.extend(["--sample_solver", args.sample_solver])
    if args.offload_model is not None:
        cmd.extend(["--offload_model", str(args.offload_model).lower()])
    if args.t5_cpu:
        cmd.append("--t5_cpu")

    env = os.environ.copy()
    repo = str(args.wan_repo.resolve())
    env["PYTHONPATH"] = repo + os.pathsep + env.get("PYTHONPATH", "")
    if args.cuda_visible_devices:
        env["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices

    start = time.time()
    with log_file.open("w", encoding="utf-8") as log:
        log.write("$ " + " ".join(cmd) + "\n\n")
        proc = subprocess.run(
            cmd,
            cwd=args.wan_repo,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
        )
    elapsed = time.time() - start

    status = "ok" if proc.returncode == 0 and out_video.exists() and out_video.stat().st_size > 0 else "failed"
    return {
        "idx": idx,
        "sample_id": row.get("sample_id", ""),
        "sample_name": row.get("sample_name", ""),
        "dataset_part": row.get("dataset_part", ""),
        "view_type": row.get("view_type", ""),
        "phase": row.get("phase", ""),
        "image": image,
        "video": str(out_video),
        "log": str(log_file),
        "seed": seed,
        "prompt_words": len(prompt.split()),
        "elapsed_sec": f"{elapsed:.2f}",
        "returncode": proc.returncode,
        "status": status,
    }


def write_report(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = [
        "idx",
        "sample_id",
        "sample_name",
        "dataset_part",
        "view_type",
        "phase",
        "status",
        "returncode",
        "seed",
        "prompt_words",
        "elapsed_sec",
        "image",
        "video",
        "log",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in keys})


def write_summary(path: Path, rows: list[dict[str, Any]], args: argparse.Namespace) -> None:
    by_status: dict[str, int] = {}
    by_view: dict[str, int] = {}
    for row in rows:
        by_status[row["status"]] = by_status.get(row["status"], 0) + 1
        view = row.get("view_type") or "(unknown)"
        by_view[view] = by_view.get(view, 0) + 1
    summary = {
        "manifest": str(args.manifest),
        "out_dir": str(args.out_dir),
        "wan_repo": str(args.wan_repo),
        "ckpt_dir": str(args.ckpt_dir),
        "rows": len(rows),
        "by_status": by_status,
        "by_view": by_view,
        "size": args.size,
        "frame_num": args.frame_num,
        "sample_steps": args.sample_steps,
        "sample_guide_scale": args.sample_guide_scale,
        "sample_shift": args.sample_shift,
        "image_policy": args.image_policy,
    }
    path.write_text(json.dumps(summary, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--wan_repo", type=Path, default=Path("/workspace/third_party/Wan2.2"))
    parser.add_argument("--ckpt_dir", type=Path, default=Path("/workspace/models/Wan2.2-TI2V-5B"))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--size", default="1280*704")
    parser.add_argument("--frame_num", type=int, default=81)
    parser.add_argument("--sample_steps", type=int, default=10)
    parser.add_argument("--sample_solver", default="unipc")
    parser.add_argument("--sample_shift", type=float, default=5.0)
    parser.add_argument("--sample_guide_scale", type=float, default=5.0)
    parser.add_argument("--base_seed", type=int, default=11)
    parser.add_argument("--image_policy", choices=["first", "middle", "last"], default="last")
    parser.add_argument("--max_prompt_words", type=int, default=120)
    parser.add_argument("--cuda_visible_devices", default=None)
    parser.add_argument("--offload_model", choices=["true", "false"], default=None)
    parser.add_argument("--t5_cpu", action="store_true")
    parser.add_argument("--skip_existing", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.manifest = args.manifest.resolve()
    args.out_dir = args.out_dir.resolve()
    args.wan_repo = args.wan_repo.resolve()
    args.ckpt_dir = args.ckpt_dir.resolve()

    rows = read_manifest(args.manifest)
    if args.offset:
        rows = rows[args.offset :]
    if args.limit is not None:
        rows = rows[: args.limit]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    video_dir = args.out_dir / "videos"
    log_dir = args.out_dir / "logs"
    video_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    print_header("WAN2.2 TI2V MANIFEST INFERENCE")
    print_kv("manifest", args.manifest)
    print_kv("out_dir", args.out_dir)
    print_kv("wan_repo", args.wan_repo)
    print_kv("ckpt_dir", args.ckpt_dir)
    print_kv("rows", len(rows))
    print_kv("size", args.size)
    print_kv("frame_num", args.frame_num)
    print_kv("sample_steps", args.sample_steps)
    print_kv("image_policy", args.image_policy)
    print_kv("cuda_visible_devices", args.cuda_visible_devices or os.environ.get("CUDA_VISIBLE_DEVICES", "(unset)"))

    report_rows: list[dict[str, Any]] = []
    for idx, row in enumerate(rows):
        sample_id = slugify(str(row.get("sample_id") or row.get("sample_name") or f"sample_{idx:06d}"))
        out_video = video_dir / f"{idx:06d}_{sample_id}.mp4"
        log_file = log_dir / f"{idx:06d}_{sample_id}.log"

        if args.skip_existing and out_video.exists() and out_video.stat().st_size > 0:
            result = {
                "idx": idx,
                "sample_id": row.get("sample_id", ""),
                "sample_name": row.get("sample_name", ""),
                "dataset_part": row.get("dataset_part", ""),
                "view_type": row.get("view_type", ""),
                "phase": row.get("phase", ""),
                "image": select_condition_image(row, args.image_policy),
                "video": str(out_video),
                "log": str(log_file),
                "seed": args.base_seed + idx if args.base_seed >= 0 else -1,
                "prompt_words": len(build_prompt(row, args.max_prompt_words).split()),
                "elapsed_sec": "0.00",
                "returncode": 0,
                "status": "skipped",
            }
        else:
            print_kv(f"[{idx + 1}/{len(rows)}] sample", sample_id)
            result = run_one(row=row, idx=idx, args=args, out_video=out_video, log_file=log_file)
            print_kv("status", f"{result['status']} | {result['elapsed_sec']} sec")
        report_rows.append(result)
        write_report(args.out_dir / "report.csv", report_rows)
        write_summary(args.out_dir / "summary.json", report_rows, args)
        if result["status"] == "failed":
            raise SystemExit(1)

    print_header("RESULT")
    by_status: dict[str, int] = {}
    for row in report_rows:
        by_status[row["status"]] = by_status.get(row["status"], 0) + 1
    print_kv("report", args.out_dir / "report.csv")
    print_kv("summary", args.out_dir / "summary.json")
    print_kv("videos", video_dir)
    print_kv("by_status", by_status)
    if by_status.get("failed", 0):
        raise SystemExit(1)


if __name__ == "__main__":
    main()

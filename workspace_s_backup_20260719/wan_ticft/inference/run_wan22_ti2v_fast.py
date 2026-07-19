"""Run Wan2.2 TI2V inference from a manifest while keeping the model loaded.

Compared with ``run_wan22_ti2v.py``, this imports the Wan2.2 pipeline directly
and reuses T5/VAE/DiT across all rows. Use it after the subprocess smoke test is
known to work in the target environment.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

import torch
import imageio
import torchvision
from PIL import Image


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
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid json: {exc}") from exc
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


def import_wan(wan_repo: Path) -> tuple[Any, Any, Any]:
    repo = str(wan_repo.resolve())
    if repo not in sys.path:
        sys.path.insert(0, repo)
    import wan
    from wan.configs import MAX_AREA_CONFIGS, SIZE_CONFIGS, WAN_CONFIGS

    return wan, WAN_CONFIGS, SIZE_CONFIGS, MAX_AREA_CONFIGS


def tensor_summary(tensor: Any) -> str:
    if tensor is None:
        return "None"
    if not torch.is_tensor(tensor):
        return f"{type(tensor).__name__}"
    return f"shape={tuple(tensor.shape)}, dtype={tensor.dtype}, device={tensor.device}"


def save_video_checked(video: torch.Tensor, save_file: Path, fps: int) -> None:
    """Save a Wan video tensor and surface failures.

    Upstream ``wan.utils.save_video`` catches and logs exceptions without raising,
    which makes batch reports ambiguous. This version mirrors the same layout but
    lets failures propagate.
    """
    if video is None or not torch.is_tensor(video):
        raise TypeError(f"expected torch.Tensor video, got {tensor_summary(video)}")
    if video.ndim != 4:
        raise ValueError(f"expected video shape [C,T,H,W], got {tuple(video.shape)}")

    tensor = video[None].clamp(-1, 1)
    frames = torch.stack(
        [
            torchvision.utils.make_grid(
                u,
                nrow=1,
                normalize=True,
                value_range=(-1, 1),
            )
            for u in tensor.unbind(2)
        ],
        dim=1,
    ).permute(1, 2, 3, 0)
    frames = (frames * 255).type(torch.uint8).cpu().numpy()

    save_file.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(str(save_file), fps=fps, codec="libx264", quality=8)
    try:
        for frame in frames:
            writer.append_data(frame)
    finally:
        writer.close()

    if not save_file.exists() or save_file.stat().st_size <= 0:
        raise RuntimeError(f"video writer produced no file: {save_file}")


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
        "seed",
        "prompt_words",
        "elapsed_sec",
        "image",
        "video",
        "video_tensor",
        "error",
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
        "offload_model": args.offload_model,
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
    parser.add_argument("--device_id", type=int, default=0)
    parser.add_argument("--offload_model", action="store_true")
    parser.add_argument("--t5_cpu", action="store_true")
    parser.add_argument("--convert_model_dtype", action="store_true")
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
    video_dir.mkdir(parents=True, exist_ok=True)

    print_header("WAN2.2 TI2V FAST MANIFEST INFERENCE")
    print_kv("manifest", args.manifest)
    print_kv("out_dir", args.out_dir)
    print_kv("wan_repo", args.wan_repo)
    print_kv("ckpt_dir", args.ckpt_dir)
    print_kv("rows", len(rows))
    print_kv("size", args.size)
    print_kv("frame_num", args.frame_num)
    print_kv("sample_steps", args.sample_steps)
    print_kv("device_id", args.device_id)
    print_kv("offload_model", args.offload_model)

    wan, WAN_CONFIGS, SIZE_CONFIGS, MAX_AREA_CONFIGS = import_wan(args.wan_repo)
    cfg = WAN_CONFIGS["ti2v-5B"]
    print_kv("model", "loading WanTI2V")
    pipe = wan.WanTI2V(
        config=cfg,
        checkpoint_dir=str(args.ckpt_dir),
        device_id=args.device_id,
        rank=0,
        t5_fsdp=False,
        dit_fsdp=False,
        use_sp=False,
        t5_cpu=args.t5_cpu,
        convert_model_dtype=args.convert_model_dtype,
    )
    print_kv("model", "loaded")

    report_rows: list[dict[str, Any]] = []
    for idx, row in enumerate(rows):
        sample_id = slugify(str(row.get("sample_id") or row.get("sample_name") or f"sample_{idx:06d}"))
        out_video = video_dir / f"{idx:06d}_{sample_id}.mp4"
        image_path = select_condition_image(row, args.image_policy)
        prompt = build_prompt(row, args.max_prompt_words)
        seed = args.base_seed + idx if args.base_seed >= 0 else -1

        result = {
            "idx": idx,
            "sample_id": row.get("sample_id", ""),
            "sample_name": row.get("sample_name", ""),
            "dataset_part": row.get("dataset_part", ""),
            "view_type": row.get("view_type", ""),
            "phase": row.get("phase", ""),
            "image": image_path,
            "video": str(out_video),
            "seed": seed,
            "prompt_words": len(prompt.split()),
            "elapsed_sec": "0.00",
            "status": "failed",
            "error": "",
            "video_tensor": "",
        }

        if args.skip_existing and out_video.exists() and out_video.stat().st_size > 0:
            result["status"] = "skipped"
            report_rows.append(result)
            continue

        print_kv(f"[{idx + 1}/{len(rows)}] sample", sample_id)
        start = time.time()
        try:
            img = Image.open(image_path).convert("RGB")
            video = pipe.generate(
                prompt,
                img=img,
                size=SIZE_CONFIGS[args.size],
                max_area=MAX_AREA_CONFIGS[args.size],
                frame_num=args.frame_num,
                shift=args.sample_shift,
                sample_solver=args.sample_solver,
                sampling_steps=args.sample_steps,
                guide_scale=args.sample_guide_scale,
                seed=seed,
                offload_model=args.offload_model,
            )
            result["video_tensor"] = tensor_summary(video)
            save_video_checked(video, out_video, fps=cfg.sample_fps)
            del video
            torch.cuda.synchronize()
            result["status"] = "ok" if out_video.exists() and out_video.stat().st_size > 0 else "failed"
        except Exception as exc:  # pragma: no cover - runtime environment
            result["error"] = repr(exc)
        result["elapsed_sec"] = f"{time.time() - start:.2f}"
        print_kv("status", f"{result['status']} | {result['elapsed_sec']} sec")

        report_rows.append(result)
        write_report(args.out_dir / "report.csv", report_rows)
        write_summary(args.out_dir / "summary.json", report_rows, args)

    print_header("RESULT")
    by_status: dict[str, int] = {}
    for row in report_rows:
        by_status[row["status"]] = by_status.get(row["status"], 0) + 1
    print_kv("report", args.out_dir / "report.csv")
    print_kv("summary", args.out_dir / "summary.json")
    print_kv("videos", video_dir)
    print_kv("by_status", by_status)


if __name__ == "__main__":
    main()

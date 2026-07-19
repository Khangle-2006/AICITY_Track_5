from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from .console import kv, section
from .frame_io import clean_output_dir, import_cv2_np, load_history_frames
from .index_io import load_test_rows, write_manifest
from .montage import make_montage
from .naming import parse_int, resolve_output_name
from .refiner_features import build_feature_config, make_refiner_arrays, refiner_in_channels
from .refiner_model import TinyResidualRefiner, import_torch
from .verify import verify_sample_output


def tensor_from_features(torch, np, features):
    return torch.from_numpy(features.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0)


def tensor_from_rgb(torch, np, img):
    return torch.from_numpy(img.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0)


def tensor_to_rgb(np, tensor):
    arr = tensor.detach().cpu().clamp(0.0, 1.0).squeeze(0).permute(1, 2, 0).numpy()
    return (arr * 255.0 + 0.5).astype(np.uint8)


def resize_frames(cv2, frames, width: int, height: int):
    return [cv2.resize(f, (width, height), interpolation=cv2.INTER_AREA) for f in frames]


def refine_frame(
    cv2,
    np,
    torch,
    model,
    device,
    frames_small,
    view_type: str,
    horizon: int,
    feature_cfg,
    history_gap: int = 1,
    phase: str = "",
):
    # Target is not used to build features for inference; pass a placeholder with
    # correct shape because make_refiner_arrays returns it unchanged.
    placeholder = frames_small[-1]
    features, _, base = make_refiner_arrays(
        cv2,
        np,
        frames_small,
        placeholder,
        view_type,
        horizon,
        feature_cfg,
        history_gap,
        phase,
    )
    x = tensor_from_features(torch, np, features).to(device)
    base_t = tensor_from_rgb(torch, np, base).to(device)
    with torch.no_grad():
        pred = model(x, base_t)
    return tensor_to_rgb(np, pred), base


def blend_rgb(np, stable, rollout, alpha: float):
    alpha = max(0.0, min(1.0, float(alpha)))
    if alpha <= 0.0:
        return stable
    if alpha >= 1.0:
        return rollout
    out = stable.astype(np.float32) * (1.0 - alpha) + rollout.astype(np.float32) * alpha
    return np.clip(out + 0.5, 0, 255).astype(np.uint8)


def process_row(cv2, np, torch, model, device, row: dict[str, str], output_name: str, args, feature_cfg) -> dict[str, Any]:
    target_n = parse_int(row.get("target_N", ""), 0)
    out_dir = Path(args.out) / output_name
    width = parse_int(row.get("frame_width", ""), 1280)
    height = parse_int(row.get("frame_height", ""), 720)

    base_row = {
        "sample_id": row.get("sample_id", ""),
        "source_folder": row.get("source_folder", ""),
        "dataset_part": row.get("dataset_part", ""),
        "view_type": row.get("view_type", ""),
        "phase": row.get("phase", ""),
        "history_K": row.get("history_K", ""),
        "target_N": target_n,
        "frame_width": width,
        "frame_height": height,
        "input_dir": row.get("input_dir", ""),
        "output_dir": str(out_dir),
        "method": "refiner",
        "frames_written": 0,
        "status": "pending",
        "notes": "",
    }

    if target_n <= 0:
        base_row["status"] = "failed"
        base_row["notes"] = "invalid target_N"
        return base_row

    frames, frame_paths = load_history_frames(cv2, row, width, height)
    if len(frames) < 1:
        base_row["status"] = "failed"
        base_row["notes"] = "missing history frames"
        return base_row

    frames_small = resize_frames(cv2, frames, args.train_width, args.train_height)
    anchor_frames_small = list(frames_small)
    history_gap = 1
    if len(frame_paths) >= 2:
        try:
            history_gap = max(1, int(frame_paths[-1].stem) - int(frame_paths[-2].stem))
        except ValueError:
            history_gap = 1

    if args.clean:
        clean_output_dir(out_dir)
    else:
        out_dir.mkdir(parents=True, exist_ok=True)

    params = [cv2.IMWRITE_PNG_COMPRESSION, args.png_compression]
    written = 0
    for i in range(target_n):
        stable_horizon = min(max(1, i + 1), args.max_infer_horizon)
        stable_small, _ = refine_frame(
            cv2,
            np,
            torch,
            model,
            device,
            anchor_frames_small,
            row.get("view_type", ""),
            stable_horizon,
            feature_cfg,
            history_gap,
            row.get("phase", ""),
        )
        pred_small = stable_small
        if args.feedback_generated and args.rollout_alpha > 0.0:
            rollout_horizon = max(1, args.rollout_step_horizon)
            rollout_small, _ = refine_frame(
                cv2,
                np,
                torch,
                model,
                device,
                frames_small,
                row.get("view_type", ""),
                rollout_horizon,
                feature_cfg,
                1,
                row.get("phase", ""),
            )
            pred_small = blend_rgb(np, stable_small, rollout_small, args.rollout_alpha)
        pred = cv2.resize(pred_small, (width, height), interpolation=cv2.INTER_LINEAR)
        out_path = out_dir / f"{i}.png"
        if cv2.imwrite(str(out_path), cv2.cvtColor(pred, cv2.COLOR_RGB2BGR), params):
            written += 1
        if args.feedback_generated:
            frames_small.append(pred_small)
            if args.rollout_history > 0 and len(frames_small) > args.rollout_history:
                frames_small = frames_small[-args.rollout_history :]

    vr = verify_sample_output(out_dir, target_n)
    base_row["frames_written"] = written
    base_row["status"] = "ok" if vr.ok and written == target_n else "bad"
    base_row["notes"] = vr.notes

    if args.make_montage and base_row["status"] == "ok":
        make_montage(cv2, np, row, frames, out_dir, Path(args.out) / "_montages" / f"{output_name}.png")

    return base_row


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_status: dict[str, int] = {}
    by_view: dict[str, int] = {}
    total_frames = 0
    for row in rows:
        by_status[row["status"]] = by_status.get(row["status"], 0) + 1
        by_view[row["view_type"]] = by_view.get(row["view_type"], 0) + 1
        total_frames += parse_int(str(row.get("frames_written", "")), 0)
    return {"rows": len(rows), "by_status": by_status, "by_view": by_view, "frames_written": total_frames}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index_dir", default="/workspace/experiments/index")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out", default="/workspace/experiments/refiner_e2e")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--clean", action="store_true", default=True)
    parser.add_argument("--no_clean", action="store_false", dest="clean")
    parser.add_argument("--make_montage", action="store_true")
    parser.add_argument("--png_compression", type=int, default=3)
    parser.add_argument("--max_infer_horizon", type=int, default=16)
    parser.add_argument("--feedback_generated", action="store_true", default=True)
    parser.add_argument("--no_feedback_generated", action="store_false", dest="feedback_generated")
    parser.add_argument("--rollout_history", type=int, default=8)
    parser.add_argument("--rollout_step_horizon", type=int, default=1)
    parser.add_argument("--rollout_alpha", type=float, default=0.35, help="Blend weight for feedback rollout prediction. 0 reproduces anchored one-shot refiner; 1 is pure rollout.")

    parser.add_argument("--train_width", type=int, default=320)
    parser.add_argument("--train_height", type=int, default=180)
    parser.add_argument("--model_width", type=int, default=48)
    parser.add_argument("--residual_scale", type=float, default=0.25)

    parser.add_argument("--flow_width", type=int, default=426)
    parser.add_argument("--flow_decay", type=float, default=0.92)
    parser.add_argument("--max_flow_scale", type=float, default=12.0)
    parser.add_argument("--light_flow_decay", type=float, default=0.80)
    parser.add_argument("--light_max_flow_scale", type=float, default=4.0)
    parser.add_argument("--region_motion_tau", type=float, default=1.5)
    parser.add_argument("--region_motion_temp", type=float, default=0.75)
    parser.add_argument("--region_max_alpha", type=float, default=0.85)
    parser.add_argument("--region_overhead_max_alpha", type=float, default=0.45)
    parser.add_argument("--region_mask_blur", type=float, default=3.0)
    parser.add_argument("--time_normalized_features", action="store_true")
    parser.add_argument("--include_context_features", action="store_true")
    parser.add_argument("--max_context_horizon", type=int, default=120)
    parser.add_argument("--max_context_history_gap", type=int, default=30)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    torch, _, _ = import_torch()
    cv2, np = import_cv2_np()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint_path = Path(args.checkpoint).expanduser()
    ckpt = torch.load(checkpoint_path, map_location=device)
    ckpt_args = ckpt.get("args", {})
    args.model_width = int(ckpt_args.get("model_width", args.model_width))
    args.residual_scale = float(ckpt_args.get("residual_scale", args.residual_scale))
    args.train_width = int(ckpt_args.get("train_width", args.train_width))
    args.train_height = int(ckpt_args.get("train_height", args.train_height))
    args.time_normalized_features = bool(
        ckpt_args.get("time_normalized_features", args.time_normalized_features)
    )
    args.include_context_features = bool(
        ckpt_args.get("include_context_features", args.include_context_features)
    )
    args.max_context_horizon = int(ckpt_args.get("max_context_horizon", args.max_context_horizon))
    args.max_context_history_gap = int(
        ckpt_args.get("max_context_history_gap", args.max_context_history_gap)
    )
    feature_cfg = build_feature_config(args)

    model = TinyResidualRefiner.build(
        in_channels=refiner_in_channels(feature_cfg),
        width=args.model_width,
        residual_scale=args.residual_scale,
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    index_dir = Path(args.index_dir).expanduser()
    out_dir = Path(args.out).expanduser()
    rows = load_test_rows(index_dir)
    if args.limit and args.limit > 0:
        rows = rows[: args.limit]

    used: set[str] = set()
    output_names = [resolve_output_name(row, used) for row in rows]

    section("INFER RESIDUAL FLOW REFINER")
    kv("index_dir", index_dir)
    kv("checkpoint", checkpoint_path)
    kv("out", out_dir)
    kv("device", device)
    kv("rows", len(rows))
    kv("size", f"{args.train_width}x{args.train_height}")
    kv("max_infer_horizon", args.max_infer_horizon)
    kv("feedback_generated", args.feedback_generated)
    if args.feedback_generated:
        kv("rollout_history", args.rollout_history)
        kv("rollout_step_horizon", args.rollout_step_horizon)
        kv("rollout_alpha", args.rollout_alpha)

    manifest_rows = []
    failures = []
    for row, output_name in zip(rows, output_names):
        item = process_row(cv2, np, torch, model, device, row, output_name, args, feature_cfg)
        manifest_rows.append(item)
        if item["status"] != "ok":
            failures.append(item)

    manifest_path = out_dir / "manifest.csv"
    write_manifest(manifest_path, manifest_rows)
    summary = summarize(manifest_rows)
    summary_path = out_dir / "summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    section("RESULT")
    kv("manifest", manifest_path)
    kv("summary", summary_path)
    kv("rows", summary["rows"])
    kv("frames_written", summary["frames_written"])
    kv("by_status", summary["by_status"])
    kv("by_view", summary["by_view"])
    kv("failures", len(failures))
    if args.make_montage:
        kv("montage_dir", out_dir / "_montages")
    if failures:
        section("FIRST FAILURES")
        for item in failures[:20]:
            print(f"  - {item['source_folder']} | {item['status']} | {item['notes']}")
        raise SystemExit(2)


if __name__ == "__main__":
    main()

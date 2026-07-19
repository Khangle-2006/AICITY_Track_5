"""Train a motion-locked high-frequency residual refiner.

Dense flow is estimated from the final observed frame to each base prediction.
The learned model operates only in that stabilized coordinate system. Its
high-pass, confidence-gated residual is inverse-warped and added to the
untouched base frame, so camera/object geometry remains owned by the base.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
from pathlib import Path

import numpy as np

from baseline.frame_io import import_cv2_np, read_rgb
from wan_ticft.evaluation.evaluate_video_reports import map_target_offsets, read_report, read_video_rgb_frames, video_fps


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def resize(cv2, image, width: int, height: int):
    if image.shape[1] == width and image.shape[0] == height:
        return image
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)


def load_base_frames(cv2, row: dict, video: Path):
    offsets = map_target_offsets(row, video_fps(cv2, video))
    frame_map = read_video_rgb_frames(cv2, video, offsets)
    return [frame_map[index] for index in offsets if index in frame_map]


def load_images(cv2, paths: list[str]):
    return [image for raw_path in paths if (image := read_rgb(cv2, Path(raw_path))) is not None]


def warp(cv2, value, flow, border_mode=None):
    height, width = flow.shape[:2]
    grid_x, grid_y = np.meshgrid(
        np.arange(width, dtype=np.float32),
        np.arange(height, dtype=np.float32),
    )
    return cv2.remap(
        value,
        grid_x + flow[..., 0],
        grid_y + flow[..., 1],
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101 if border_mode is None else border_mode,
    )


def dense_flow(cv2, source_rgb, destination_rgb):
    source = cv2.cvtColor(source_rgb, cv2.COLOR_RGB2GRAY)
    destination = cv2.cvtColor(destination_rgb, cv2.COLOR_RGB2GRAY)
    return cv2.calcOpticalFlowFarneback(source, destination, None, 0.5, 5, 25, 4, 7, 1.5, 0)


def flow_pair(cv2, anchor_rgb, base_rgb, consistency_px: float):
    anchor_to_base = dense_flow(cv2, anchor_rgb, base_rgb)
    base_to_anchor = dense_flow(cv2, base_rgb, anchor_rgb)
    backward_on_anchor = warp(cv2, base_to_anchor, anchor_to_base, cv2.BORDER_CONSTANT)
    forward_on_base = warp(cv2, anchor_to_base, base_to_anchor, cv2.BORDER_CONSTANT)
    anchor_error = ((anchor_to_base + backward_on_anchor) ** 2).sum(axis=2) ** 0.5
    base_error = ((base_to_anchor + forward_on_base) ** 2).sum(axis=2) ** 0.5
    anchor_confidence = np.exp(-((anchor_error / consistency_px) ** 2))[..., None].astype(np.float32)
    base_confidence = np.exp(-((base_error / consistency_px) ** 2))[..., None].astype(np.float32)
    return anchor_to_base, base_to_anchor, anchor_confidence, base_confidence


def highpass(cv2, value, sigma: float):
    return value - cv2.GaussianBlur(value, (0, 0), sigma)


def prepare_frame(cv2, np, anchor, base, target, time_value: float, sigma: float, consistency_px: float):
    anchor_f = anchor.astype(np.float32) / 255.0
    base_f = base.astype(np.float32) / 255.0
    target_f = target.astype(np.float32) / 255.0
    anchor_to_base, base_to_anchor, anchor_confidence, base_confidence = flow_pair(
        cv2, anchor, base, consistency_px
    )
    aligned_base = warp(cv2, base_f, anchor_to_base)
    target_residual_anchor = warp(cv2, target_f - base_f, anchor_to_base)
    target_detail = highpass(cv2, target_residual_anchor, sigma) * anchor_confidence
    height, width = base.shape[:2]
    normalized_flow = anchor_to_base.copy().astype(np.float32)
    normalized_flow[..., 0] /= max(width, 1)
    normalized_flow[..., 1] /= max(height, 1)
    model_input = np.concatenate(
        [
            aligned_base,
            anchor_f,
            highpass(cv2, aligned_base, sigma),
            highpass(cv2, anchor_f, sigma),
            normalized_flow,
            anchor_confidence,
            np.full((height, width, 1), time_value, dtype=np.float32),
        ],
        axis=2,
    )
    return {
        "x": model_input.transpose(2, 0, 1).astype(np.float16),
        "detail": target_detail.transpose(2, 0, 1).astype(np.float16),
        "confidence": anchor_confidence.transpose(2, 0, 1).astype(np.float16),
        "base_to_anchor": base_to_anchor.astype(np.float32),
        "base_confidence": base_confidence.astype(np.float32),
    }


def prepare_sequence(cv2, np, row: dict, video: Path, width: int, height: int, sigma: float, consistency_px: float):
    bases = load_base_frames(cv2, row, video)
    targets = load_images(cv2, (row.get("frames") or {}).get("target", []))
    conditions = load_images(cv2, (row.get("frames") or {}).get("condition", []))
    if len(bases) != len(targets) or len(bases) < 2 or not conditions:
        return None
    anchor = resize(cv2, conditions[-1], width, height)
    bases = [resize(cv2, image, width, height) for image in bases]
    targets = [resize(cv2, image, width, height) for image in targets]
    prepared = []
    for index, (base, target) in enumerate(zip(bases, targets)):
        time_value = index / float(max(1, len(bases) - 1))
        prepared.append(prepare_frame(cv2, np, anchor, base, target, time_value, sigma, consistency_px))
    return {"sample_id": row["sample_id"], "frames": prepared}


def build_model(torch, nn, F, channels: int, width: int, residual_scale: float, sigma_kernel: int):
    class Block(nn.Module):
        def __init__(self, dilation: int):
            super().__init__()
            self.conv1 = nn.Conv2d(width, width, 3, padding=dilation, dilation=dilation)
            self.conv2 = nn.Conv2d(width, width, 3, padding=dilation, dilation=dilation)
            self.norm1 = nn.GroupNorm(8, width)
            self.norm2 = nn.GroupNorm(8, width)

        def forward(self, value):
            update = F.silu(self.norm1(self.conv1(value)))
            update = self.norm2(self.conv2(update))
            return F.silu(value + update)

    class FlowFrozenDetailRefiner(nn.Module):
        def __init__(self):
            super().__init__()
            self.stem = nn.Sequential(nn.Conv2d(channels, width, 3, padding=1), nn.SiLU())
            self.blocks = nn.Sequential(*[Block(dilation) for dilation in (1, 2, 3, 1, 2, 3)])
            self.head = nn.Conv2d(width, 3, 3, padding=1)

        def forward(self, value):
            confidence = value[:, 14:15]
            raw = torch.tanh(self.head(self.blocks(self.stem(value))))
            low = F.avg_pool2d(raw, kernel_size=sigma_kernel, stride=1, padding=sigma_kernel // 2)
            return (raw - low) * residual_scale * confidence

    return FlowFrozenDetailRefiner()


def laplacian(torch, F, value):
    kernel = torch.tensor(
        [[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]],
        device=value.device,
        dtype=value.dtype,
    ).view(1, 1, 3, 3)
    return F.conv2d(value, kernel.repeat(value.shape[1], 1, 1, 1), padding=1, groups=value.shape[1])


def weighted_mean(value, weight):
    return (value * weight).sum() / weight.expand_as(value).sum().clamp_min(1.0)


def sample_batch(torch, sequences, batch_size: int, patch: int, rng):
    x0s, x1s, y0s, y1s, w0s, w1s = [], [], [], [], [], []
    for _ in range(batch_size):
        sequence = rng.choice(sequences)["frames"]
        index = rng.randrange(0, len(sequence) - 1)
        _, height, width = sequence[0]["x"].shape
        if height < patch or width < patch:
            raise RuntimeError("training image is smaller than patch size")
        top = rng.randrange(0, height - patch + 1)
        left = rng.randrange(0, width - patch + 1)
        region = (slice(None), slice(top, top + patch), slice(left, left + patch))
        first, second = sequence[index], sequence[index + 1]
        x0s.append(torch.from_numpy(first["x"][region]).float())
        x1s.append(torch.from_numpy(second["x"][region]).float())
        y0s.append(torch.from_numpy(first["detail"][region]).float())
        y1s.append(torch.from_numpy(second["detail"][region]).float())
        w0s.append(torch.from_numpy(first["confidence"][region]).float())
        w1s.append(torch.from_numpy(second["confidence"][region]).float())
    return tuple(torch.stack(values) for values in (x0s, x1s, y0s, y1s, w0s, w1s))


def write_video(cv2, frames, path: Path, fps: float):
    path.parent.mkdir(parents=True, exist_ok=True)
    height, width = frames[0].shape[:2]
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"FFV1"), fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"cannot create {path}")
    for frame in [frames[0], *frames]:
        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    writer.release()


def infer_sequence(cv2, np, torch, model, row: dict, base_video: Path, device, args):
    bases = load_base_frames(cv2, row, base_video)
    conditions = load_images(cv2, (row.get("frames") or {}).get("condition", []))
    if not bases or not conditions:
        return [], []
    height, width = bases[0].shape[:2]
    anchor = resize(cv2, conditions[-1], width, height)
    output, diagnostics = [], []
    for index, base in enumerate(bases):
        base = resize(cv2, base, width, height)
        prepared = prepare_frame(
            cv2,
            np,
            anchor,
            base,
            base,
            index / float(max(1, len(bases) - 1)),
            args.highpass_sigma,
            args.consistency_px,
        )
        value = torch.from_numpy(prepared["x"]).float().unsqueeze(0).to(device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            detail_anchor = model(value)[0].float().cpu().permute(1, 2, 0).numpy()
        detail_base = warp(cv2, detail_anchor, prepared["base_to_anchor"]) * prepared["base_confidence"]
        base_f = base.astype(np.float32) / 255.0
        prediction = np.clip(base_f + detail_base, 0.0, 1.0)
        output.append((prediction * 255.0 + 0.5).astype(np.uint8))
        diagnostics.append(
            {
                "frame": index,
                "mean_abs_residual": float(np.mean(np.abs(detail_base))),
                "mean_confidence": float(np.mean(prepared["base_confidence"])),
            }
        )
    return output, diagnostics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train_manifest", type=Path, required=True)
    parser.add_argument("--train_report", type=Path, required=True)
    parser.add_argument("--eval_manifest", type=Path, required=True)
    parser.add_argument("--eval_report", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--train_limit", type=int, default=16)
    parser.add_argument("--eval_limit", type=int, default=24)
    parser.add_argument("--train_width", type=int, default=640)
    parser.add_argument("--train_height", type=int, default=360)
    parser.add_argument("--patch", type=int, default=160)
    parser.add_argument("--batch_size", type=int, default=6)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--steps_per_epoch", type=int, default=80)
    parser.add_argument("--model_width", type=int, default=48)
    parser.add_argument("--residual_scale", type=float, default=0.035)
    parser.add_argument("--highpass_sigma", type=float, default=2.0)
    parser.add_argument("--highpass_kernel", type=int, default=9)
    parser.add_argument("--consistency_px", type=float, default=2.5)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--lambda_edge", type=float, default=0.15)
    parser.add_argument("--lambda_temporal", type=float, default=0.20)
    parser.add_argument("--lambda_energy", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=20260621)
    parser.add_argument("--clean", action="store_true")
    args = parser.parse_args()
    if args.highpass_kernel % 2 != 1:
        raise ValueError("highpass_kernel must be odd")

    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    cv2, np = import_cv2_np()
    if args.clean and args.out_dir.exists():
        shutil.rmtree(args.out_dir)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "args.json").write_text(json.dumps(vars(args), default=str, indent=2), encoding="utf-8")

    train_rows = [row for row in read_jsonl(args.train_manifest) if row.get("view_type") == "bdd"][: args.train_limit]
    train_report = read_report(args.train_report)
    sequences = []
    for position, row in enumerate(train_rows, start=1):
        video = train_report.get(row["sample_id"])
        sequence = (
            prepare_sequence(
                cv2,
                np,
                row,
                video,
                args.train_width,
                args.train_height,
                args.highpass_sigma,
                args.consistency_px,
            )
            if video
            else None
        )
        if sequence is not None:
            sequences.append(sequence)
        print(f"prepare train [{position}/{len(train_rows)}] {row['sample_id']} | {'ok' if sequence else 'skip'}", flush=True)
    if len(sequences) != args.train_limit:
        raise RuntimeError(f"expected {args.train_limit} train sequences, found {len(sequences)}")

    device = torch.device("cuda")
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    model = build_model(torch, nn, F, 16, args.model_width, args.residual_scale, args.highpass_kernel).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        totals = {"loss": 0.0, "detail": 0.0, "edge": 0.0, "temporal": 0.0, "energy": 0.0}
        for _ in range(args.steps_per_epoch):
            values = sample_batch(torch, sequences, args.batch_size, args.patch, rng)
            x0, x1, y0, y1, w0, w1 = [value.to(device) for value in values]
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                p0 = model(x0)
                p1 = model(x1)
                detail = 0.5 * (
                    weighted_mean((p0 - y0).abs(), 0.1 + 0.9 * w0)
                    + weighted_mean((p1 - y1).abs(), 0.1 + 0.9 * w1)
                )
                edge = 0.5 * (
                    weighted_mean((laplacian(torch, F, p0) - laplacian(torch, F, y0)).abs(), 0.1 + 0.9 * w0)
                    + weighted_mean((laplacian(torch, F, p1) - laplacian(torch, F, y1)).abs(), 0.1 + 0.9 * w1)
                )
                temporal_weight = 0.1 + 0.9 * torch.minimum(w0, w1)
                temporal = weighted_mean(((p1 - p0) - (y1 - y0)).abs(), temporal_weight)
                energy = 0.5 * (p0.abs().mean() + p1.abs().mean())
                loss = detail + args.lambda_edge * edge + args.lambda_temporal * temporal + args.lambda_energy * energy
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            for name, value in (("loss", loss), ("detail", detail), ("edge", edge), ("temporal", temporal), ("energy", energy)):
                totals[name] += float(value.detach().cpu())
        log_row = {"epoch": epoch, **{name: value / args.steps_per_epoch for name, value in totals.items()}}
        history.append(log_row)
        print(" ".join([f"epoch={epoch}"] + [f"{name}={value:.6f}" for name, value in log_row.items() if name != "epoch"]), flush=True)

    torch.save({"model": model.state_dict(), "args": vars(args)}, args.out_dir / "flow_frozen_detail_refiner.pt")
    with (args.out_dir / "history.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=history[0].keys())
        writer.writeheader()
        writer.writerows(history)

    eval_rows = [row for row in read_jsonl(args.eval_manifest) if row.get("view_type") == "bdd"][: args.eval_limit]
    eval_report = read_report(args.eval_report)
    report_rows, diagnostic_rows = [], []
    model.eval()
    with torch.no_grad():
        for position, row in enumerate(eval_rows, start=1):
            base_video = eval_report.get(row["sample_id"])
            if base_video is None:
                continue
            output, diagnostics = infer_sequence(cv2, np, torch, model, row, base_video, device, args)
            if not output:
                continue
            path = args.out_dir / "videos" / f"{row['sample_id']}.avi"
            write_video(cv2, output, path, video_fps(cv2, base_video) or 30.0)
            report_rows.append({"sample_id": row["sample_id"], "video": str(path), "status": "ok", "future_frames": len(output)})
            for diagnostic in diagnostics:
                diagnostic_rows.append({"sample_id": row["sample_id"], **diagnostic})
            print(f"infer [{position}/{len(eval_rows)}] {row['sample_id']} | ok", flush=True)
    if not report_rows:
        raise RuntimeError("no evaluation outputs")
    with (args.out_dir / "report.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=report_rows[0].keys())
        writer.writeheader()
        writer.writerows(report_rows)
    with (args.out_dir / "diagnostics.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=diagnostic_rows[0].keys())
        writer.writeheader()
        writer.writerows(diagnostic_rows)


if __name__ == "__main__":
    main()

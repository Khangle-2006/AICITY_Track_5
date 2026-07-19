from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
from pathlib import Path

from baseline.frame_io import import_cv2_np, read_rgb
from wan_ticft.evaluation.evaluate_video_reports import map_target_offsets, read_report, read_video_rgb_frames, video_fps
from wan_ticft.evaluation.train_metric_residual_adapter import ssim_loss


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


def load_targets(cv2, row: dict):
    return [image for path in (row.get("frames") or {}).get("target", []) if (image := read_rgb(cv2, Path(path))) is not None]


def load_train_sequence(cv2, row: dict, video: Path, width: int, height: int):
    bases = load_base_frames(cv2, row, video)
    targets = load_targets(cv2, row)
    if len(bases) != len(targets) or len(bases) < 2:
        return None
    bases = [resize(cv2, image, width, height) for image in bases]
    targets = [resize(cv2, image, width, height) for image in targets]
    return {"sample_id": row["sample_id"], "base": bases, "target": targets}


def to_tensor(torch, np, image):
    return torch.from_numpy(image.astype(np.float32) / 255.0).permute(2, 0, 1)


def highpass(cv2, image):
    value = image.astype("float32") / 255.0
    return value - cv2.GaussianBlur(value, (0, 0), 2.0)


def make_input(torch, np, cv2, frames, index: int, time_value: float):
    previous = frames[max(0, index - 1)]
    current = frames[index]
    following = frames[min(len(frames) - 1, index + 1)]
    values = [
        previous.astype(np.float32) / 255.0,
        current.astype(np.float32) / 255.0,
        following.astype(np.float32) / 255.0,
        highpass(cv2, current),
        np.full((*current.shape[:2], 1), time_value, dtype=np.float32),
    ]
    return torch.from_numpy(np.concatenate(values, axis=2)).permute(2, 0, 1).float()


def build_model(torch, nn, F, channels: int, width: int, residual_scale: float):
    class Block(nn.Module):
        def __init__(self, dilation: int):
            super().__init__()
            self.conv1 = nn.Conv2d(width, width, 3, padding=dilation, dilation=dilation)
            self.conv2 = nn.Conv2d(width, width, 3, padding=dilation, dilation=dilation)
            self.norm1 = nn.GroupNorm(8, width)
            self.norm2 = nn.GroupNorm(8, width)

        def forward(self, x):
            value = F.silu(self.norm1(self.conv1(x)))
            value = self.norm2(self.conv2(value))
            return F.silu(x + value)

    class DetailRefiner(nn.Module):
        def __init__(self):
            super().__init__()
            self.stem = nn.Sequential(nn.Conv2d(channels, width, 3, padding=1), nn.SiLU())
            self.blocks = nn.Sequential(*[Block(dilation) for dilation in (1, 2, 3, 1, 2, 3)])
            self.head = nn.Conv2d(width, 3, 3, padding=1)

        def forward(self, x):
            base = x[:, 3:6]
            raw = torch.tanh(self.head(self.blocks(self.stem(x))))
            low = F.avg_pool2d(raw, kernel_size=9, stride=1, padding=4)
            detail = (raw - low) * residual_scale
            return (base + detail).clamp(0.0, 1.0), detail

    return DetailRefiner()


def laplacian(torch, F, value):
    kernel = torch.tensor(
        [[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]],
        device=value.device,
        dtype=value.dtype,
    ).view(1, 1, 3, 3)
    kernel = kernel.repeat(value.shape[1], 1, 1, 1)
    return F.conv2d(value, kernel, padding=1, groups=value.shape[1])


def sample_batch(torch, np, cv2, sequences, batch_size: int, patch: int, rng):
    x0s, x1s, y0s, y1s = [], [], [], []
    for _ in range(batch_size):
        sequence = rng.choice(sequences)
        count = len(sequence["base"])
        index = rng.randrange(0, count - 1)
        height, width = sequence["base"][0].shape[:2]
        top = rng.randrange(0, max(1, height - patch + 1))
        left = rng.randrange(0, max(1, width - patch + 1))
        bottom = min(height, top + patch)
        right = min(width, left + patch)
        if bottom - top != patch or right - left != patch:
            raise RuntimeError("training image is smaller than patch size")
        t0 = index / float(max(1, count - 1))
        t1 = (index + 1) / float(max(1, count - 1))
        x0s.append(make_input(torch, np, cv2, sequence["base"], index, t0)[:, top:bottom, left:right])
        x1s.append(make_input(torch, np, cv2, sequence["base"], index + 1, t1)[:, top:bottom, left:right])
        y0s.append(to_tensor(torch, np, sequence["target"][index])[:, top:bottom, left:right])
        y1s.append(to_tensor(torch, np, sequence["target"][index + 1])[:, top:bottom, left:right])
    return torch.stack(x0s), torch.stack(x1s), torch.stack(y0s), torch.stack(y1s)


def write_video(cv2, frames, path: Path, fps: float):
    path.parent.mkdir(parents=True, exist_ok=True)
    height, width = frames[0].shape[:2]
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"FFV1"), fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"cannot create {path}")
    for frame in [frames[0], *frames]:
        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    writer.release()


def main() -> None:
    parser = argparse.ArgumentParser(description="Patch-trained full-resolution temporal detail refiner.")
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
    parser.add_argument("--residual_scale", type=float, default=0.04)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--lambda_ssim", type=float, default=0.20)
    parser.add_argument("--lambda_edge", type=float, default=0.15)
    parser.add_argument("--lambda_temporal", type=float, default=0.10)
    parser.add_argument("--lambda_residual", type=float, default=0.04)
    parser.add_argument("--seed", type=int, default=20260620)
    parser.add_argument("--clean", action="store_true")
    args = parser.parse_args()

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
        sequence = load_train_sequence(cv2, row, video, args.train_width, args.train_height) if video else None
        if sequence is not None:
            sequences.append(sequence)
        print(f"load train [{position}/{len(train_rows)}] {row['sample_id']} | {'ok' if sequence else 'skip'}", flush=True)
    if len(sequences) != args.train_limit:
        raise RuntimeError(f"expected {args.train_limit} train sequences, found {len(sequences)}")

    device = torch.device("cuda")
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    model = build_model(torch, nn, F, channels=13, width=args.model_width, residual_scale=args.residual_scale).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        totals = {"loss": 0.0, "l1": 0.0, "ssim": 0.0, "edge": 0.0, "temporal": 0.0, "residual": 0.0}
        for _ in range(args.steps_per_epoch):
            x0, x1, y0, y1 = sample_batch(torch, np, cv2, sequences, args.batch_size, args.patch, rng)
            x0, x1, y0, y1 = [value.to(device) for value in (x0, x1, y0, y1)]
            p0, r0 = model(x0)
            p1, r1 = model(x1)
            l1 = 0.5 * ((p0 - y0).abs().mean() + (p1 - y1).abs().mean())
            sl = 0.5 * (ssim_loss(torch, F, p0, y0) + ssim_loss(torch, F, p1, y1))
            edge = 0.5 * ((laplacian(torch, F, p0) - laplacian(torch, F, y0)).abs().mean() + (laplacian(torch, F, p1) - laplacian(torch, F, y1)).abs().mean())
            temporal = ((p1 - p0) - (y1 - y0)).abs().mean()
            residual = 0.5 * (r0.abs().mean() + r1.abs().mean())
            loss = l1 + args.lambda_ssim * sl + args.lambda_edge * edge + args.lambda_temporal * temporal + args.lambda_residual * residual
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            for name, value in (("loss", loss), ("l1", l1), ("ssim", sl), ("edge", edge), ("temporal", temporal), ("residual", residual)):
                totals[name] += float(value.detach().cpu())
        row = {"epoch": epoch, **{name: value / args.steps_per_epoch for name, value in totals.items()}}
        history.append(row)
        print(" ".join([f"epoch={epoch}"] + [f"{name}={value:.6f}" for name, value in row.items() if name != "epoch"]), flush=True)

    torch.save({"model": model.state_dict(), "args": vars(args)}, args.out_dir / "highres_detail_refiner.pt")
    with (args.out_dir / "history.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=history[0].keys())
        writer.writeheader()
        writer.writerows(history)

    eval_rows = [row for row in read_jsonl(args.eval_manifest) if row.get("view_type") == "bdd"][: args.eval_limit]
    eval_report = read_report(args.eval_report)
    report_rows = []
    model.eval()
    with torch.no_grad():
        for position, row in enumerate(eval_rows, start=1):
            base_video = eval_report.get(row["sample_id"])
            if base_video is None:
                continue
            bases = load_base_frames(cv2, row, base_video)
            output = []
            for index in range(len(bases)):
                time_value = index / float(max(1, len(bases) - 1))
                x = make_input(torch, np, cv2, bases, index, time_value).unsqueeze(0).to(device)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    prediction, _ = model(x)
                image = prediction[0].float().cpu().clamp(0, 1).permute(1, 2, 0).numpy()
                output.append((image * 255.0 + 0.5).astype(np.uint8))
            path = args.out_dir / "videos" / f"{row['sample_id']}.avi"
            write_video(cv2, output, path, video_fps(cv2, base_video) or 30.0)
            report_rows.append({"sample_id": row["sample_id"], "video": str(path), "status": "ok", "future_frames": len(output)})
            print(f"infer [{position}/{len(eval_rows)}] {row['sample_id']} | ok", flush=True)
    with (args.out_dir / "report.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=report_rows[0].keys())
        writer.writeheader()
        writer.writerows(report_rows)


if __name__ == "__main__":
    main()

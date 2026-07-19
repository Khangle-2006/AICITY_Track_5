from __future__ import annotations

import argparse
import csv
import json
import math
import random
import shutil
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from wan_ticft.evaluation.train_supervised_residual_gate import (
    gradient_loss,
    load_examples,
    read_jsonl,
    write_video,
)
from wan_ticft.evaluation.evaluate_video_reports import read_report


class ResidualBlock(nn.Module):
    def __init__(self, width: int):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(width, width, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(width, width, 3, padding=1),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return F.silu(value + self.body(value))


class ConvexBlend(nn.Module):
    def __init__(self, width: int, blocks: int, init_alpha: float):
        super().__init__()
        init_alpha = min(max(init_alpha, 1e-4), 1.0 - 1e-4)
        self.stem = nn.Sequential(nn.Conv2d(9, width, 3, padding=1), nn.SiLU())
        self.blocks = nn.Sequential(*[ResidualBlock(width) for _ in range(blocks)])
        self.head = nn.Conv2d(width, 1, 3, padding=1)
        nn.init.zeros_(self.head.weight)
        nn.init.constant_(self.head.bias, math.log(init_alpha / (1.0 - init_alpha)))

    def forward(self, repeat: torch.Tensor, depth: torch.Tensor, time_value: torch.Tensor) -> torch.Tensor:
        residual = depth - repeat
        magnitude = residual.abs().mean(dim=1, keepdim=True)
        repeat_luma = repeat.mean(dim=1, keepdim=True)
        time_plane = time_value[:, :, None, None].expand(-1, -1, repeat.shape[2], repeat.shape[3])
        features = torch.cat([repeat, residual, magnitude, repeat_luma, time_plane], dim=1)
        return torch.sigmoid(self.head(self.blocks(self.stem(features))))


def tensors_from_crop(example: dict[str, object], top: int, left: int, patch: int):
    tensors = []
    for key in ("base", "candidate", "target"):
        crop = example[key][top : top + patch, left : left + patch]  # type: ignore[index]
        tensors.append(torch.from_numpy(crop.transpose(2, 0, 1).copy()).float() / 255.0)
    tensors.append(torch.tensor([float(example["time"])], dtype=torch.float32))
    return tuple(tensors)


def sample_batch(examples: list[dict[str, object]], batch_size: int, patch: int, rng: random.Random):
    values = [[], [], [], []]
    for _ in range(batch_size):
        example = rng.choice(examples)
        height, width = example["base"].shape[:2]  # type: ignore[index]
        top = rng.randrange(0, height - patch + 1)
        left = rng.randrange(0, width - patch + 1)
        for destination, tensor in zip(values, tensors_from_crop(example, top, left, patch)):
            destination.append(tensor)
    return tuple(torch.stack(items) for items in values)


@torch.inference_mode()
def infer_report(
    model: ConvexBlend,
    manifest: Path,
    repeat_csv: Path,
    depth_csv: Path,
    view_type: str,
    out_dir: Path,
    device: torch.device,
) -> None:
    repeat_report = read_report(repeat_csv)
    depth_report = read_report(depth_csv)
    rows = [row for row in read_jsonl(manifest) if row.get("view_type") == view_type]
    report_rows = []
    alpha_rows = []
    for position, row in enumerate(rows, start=1):
        sample_id = row["sample_id"]
        repeat_video = repeat_report.get(sample_id)
        depth_video = depth_report.get(sample_id)
        if repeat_video is None or depth_video is None:
            print(f"eval [{position}/{len(rows)}] {sample_id} | skipped", flush=True)
            continue
        cap_repeat = cv2.VideoCapture(str(repeat_video))
        cap_depth = cv2.VideoCapture(str(depth_video))
        fps = float(cap_depth.get(cv2.CAP_PROP_FPS) or cap_repeat.get(cv2.CAP_PROP_FPS) or 30.0)
        frame_count = min(
            int(cap_repeat.get(cv2.CAP_PROP_FRAME_COUNT) or 0),
            int(cap_depth.get(cv2.CAP_PROP_FRAME_COUNT) or 0),
        )
        outputs: list[np.ndarray] = []
        alpha_means = []
        for frame_index in range(frame_count):
            ok_repeat, repeat_bgr = cap_repeat.read()
            ok_depth, depth_bgr = cap_depth.read()
            if not ok_repeat or not ok_depth:
                break
            repeat = cv2.cvtColor(repeat_bgr, cv2.COLOR_BGR2RGB)
            depth = cv2.cvtColor(depth_bgr, cv2.COLOR_BGR2RGB)
            if repeat.shape[:2] != depth.shape[:2]:
                repeat = cv2.resize(repeat, (depth.shape[1], depth.shape[0]), interpolation=cv2.INTER_AREA)
            repeat_tensor = torch.from_numpy(repeat.transpose(2, 0, 1).copy()).float()[None].to(device) / 255.0
            depth_tensor = torch.from_numpy(depth.transpose(2, 0, 1).copy()).float()[None].to(device) / 255.0
            time_tensor = torch.tensor([[frame_index / float(max(1, frame_count - 1))]], device=device)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                alpha = model(repeat_tensor, depth_tensor, time_tensor)
                prediction = alpha * depth_tensor + (1.0 - alpha) * repeat_tensor
            alpha_means.append(float(alpha.float().mean().item()))
            image = prediction[0].clamp(0, 1).float().cpu().permute(1, 2, 0).numpy()
            outputs.append((image * 255.0 + 0.5).astype(np.uint8))
        cap_repeat.release()
        cap_depth.release()
        if len(outputs) != frame_count or not outputs:
            raise RuntimeError(f"{sample_id}: decoded {len(outputs)}/{frame_count}")
        video_path = out_dir / "videos" / f"{sample_id}.avi"
        write_video(outputs, video_path, fps)
        report_rows.append({"sample_id": sample_id, "video": str(video_path), "status": "ok", "future_frames": len(outputs)})
        alpha_rows.append(
            {
                "sample_id": sample_id,
                "alpha_mean": float(np.mean(alpha_means)),
                "alpha_min": float(np.min(alpha_means)),
                "alpha_max": float(np.max(alpha_means)),
            }
        )
        print(f"eval [{position}/{len(rows)}] {sample_id} | frames={len(outputs)} alpha={np.mean(alpha_means):.3f}", flush=True)
    with (out_dir / "report.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=report_rows[0].keys())
        writer.writeheader()
        writer.writerows(report_rows)
    with (out_dir / "alpha_stats.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=alpha_rows[0].keys())
        writer.writeheader()
        writer.writerows(alpha_rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a bounded repeat/depth convex blend.")
    parser.add_argument("--train_manifest", type=Path, required=True)
    parser.add_argument("--train_repeat_report", type=Path, required=True)
    parser.add_argument("--train_depth_report", type=Path, required=True)
    parser.add_argument("--eval_manifest", type=Path, required=True)
    parser.add_argument("--eval_repeat_report", type=Path, required=True)
    parser.add_argument("--eval_depth_report", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--view_type", default="bdd")
    parser.add_argument("--train_limit", type=int, default=48)
    parser.add_argument("--train_width", type=int, default=640)
    parser.add_argument("--train_height", type=int, default=360)
    parser.add_argument("--patch", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--steps_per_epoch", type=int, default=180)
    parser.add_argument("--model_width", type=int, default=32)
    parser.add_argument("--model_blocks", type=int, default=5)
    parser.add_argument("--init_alpha", type=float, default=0.90, help="Initial weight on depth candidate.")
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--lambda_edge", type=float, default=0.08)
    parser.add_argument("--lambda_depth_prior", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=20260623)
    parser.add_argument("--clean", action="store_true")
    args = parser.parse_args()

    if args.clean and args.out_dir.exists():
        shutil.rmtree(args.out_dir)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "args.json").write_text(json.dumps(vars(args), default=str, indent=2), encoding="utf-8")

    examples = load_examples(
        args.train_manifest,
        args.train_repeat_report,
        args.train_depth_report,
        args.view_type,
        args.train_limit,
        args.train_width,
        args.train_height,
    )
    device = torch.device("cuda")
    model = ConvexBlend(args.model_width, args.model_blocks, args.init_alpha).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    rng = random.Random(args.seed)
    history = []
    model.train()
    for epoch in range(args.epochs):
        losses = []
        for _ in range(args.steps_per_epoch):
            repeat, depth, target, time_value = sample_batch(examples, args.batch_size, args.patch, rng)
            repeat, depth, target, time_value = [value.to(device) for value in (repeat, depth, target, time_value)]
            alpha = model(repeat, depth, time_value)
            prediction = alpha * depth + (1.0 - alpha) * repeat
            pixel = torch.sqrt((prediction - target) ** 2 + 1e-6).mean()
            edge = gradient_loss(prediction, target)
            depth_prior = ((1.0 - alpha) ** 2).mean()
            loss = pixel + args.lambda_edge * edge + args.lambda_depth_prior * depth_prior
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(
                (
                    float(loss.item()),
                    float(pixel.item()),
                    float(edge.item()),
                    float(depth_prior.item()),
                    float(alpha.mean().item()),
                )
            )
        means = np.mean(np.asarray(losses), axis=0)
        row = {
            "epoch": epoch + 1,
            "loss": means[0],
            "pixel": means[1],
            "edge": means[2],
            "depth_prior": means[3],
            "alpha_mean": means[4],
        }
        history.append(row)
        print(
            "epoch={epoch} loss={loss:.6f} pixel={pixel:.6f} edge={edge:.6f} depth_prior={depth_prior:.6f} alpha={alpha_mean:.4f}".format(
                **row
            ),
            flush=True,
        )
    torch.save({"model": model.state_dict(), "args": vars(args), "examples": len(examples)}, args.out_dir / "model.pt")
    with (args.out_dir / "history.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=history[0].keys())
        writer.writeheader()
        writer.writerows(history)
    model.eval()
    infer_report(model, args.eval_manifest, args.eval_repeat_report, args.eval_depth_report, args.view_type, args.out_dir, device)
    print(f"examples={len(examples)}")
    print(f"output={args.out_dir}")


if __name__ == "__main__":
    main()

"""Train a bounded supervised gate over a frozen depth-renderer residual."""

from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from baseline.frame_io import import_cv2_np, read_rgb
from wan_ticft.evaluation.evaluate_video_reports import (
    map_target_offsets,
    read_report,
    read_video_rgb_frames,
    video_fps,
)


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_examples(cv2_module, manifest: Path, base_csv: Path, candidate_csv: Path, limit: int, width: int, height: int):
    base_report = read_report(base_csv)
    candidate_report = read_report(candidate_csv)
    examples = []
    rows = [row for row in read_jsonl(manifest) if row.get("view_type") == "wts_vehicle_or_ip_camera"]
    for row in rows[:limit]:
        sample_id = row["sample_id"]
        base_video = base_report.get(sample_id)
        candidate_video = candidate_report.get(sample_id)
        if base_video is None or candidate_video is None:
            continue
        offsets = map_target_offsets(row, video_fps(cv2_module, base_video))
        base_map = read_video_rgb_frames(cv2_module, base_video, offsets)
        candidate_map = read_video_rgb_frames(cv2_module, candidate_video, offsets)
        targets = [read_rgb(cv2_module, Path(path)) for path in (row.get("frames") or {}).get("target", [])]
        if len(targets) != len(offsets) or any(target is None for target in targets):
            continue
        max_offset = max(offsets)
        for offset, target in zip(offsets, targets):
            if offset not in base_map or offset not in candidate_map:
                continue
            resize = lambda image: cv2_module.resize(image, (width, height), interpolation=cv2_module.INTER_AREA)
            examples.append(
                {
                    "sample_id": sample_id,
                    "base": resize(base_map[offset]),
                    "candidate": resize(candidate_map[offset]),
                    "target": resize(target),
                    "time": offset / float(max(1, max_offset)),
                }
            )
    if not examples:
        raise RuntimeError("no aligned supervised examples")
    return examples


class ResidualBlock(nn.Module):
    def __init__(self, width: int):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(width, width, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(width, width, 3, padding=1),
        )

    def forward(self, value):
        return F.silu(value + self.body(value))


class SupervisedDepthGate(nn.Module):
    def __init__(self, width: int):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(8, width, 3, padding=1), nn.SiLU())
        self.blocks = nn.Sequential(*[ResidualBlock(width) for _ in range(4)])
        self.head = nn.Conv2d(width, 1, 3, padding=1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, base, candidate, time_value):
        residual = candidate - base
        magnitude = residual.abs().mean(dim=1, keepdim=True)
        time_plane = time_value[:, :, None, None].expand(-1, -1, base.shape[2], base.shape[3])
        features = torch.cat([base, residual, magnitude, time_plane], dim=1)
        raw = self.head(self.blocks(self.stem(features)))
        return 1.0 + 0.5 * torch.tanh(raw)


def tensors_from_crop(example, top: int, left: int, patch: int):
    arrays = []
    for key in ("base", "candidate", "target"):
        crop = example[key][top : top + patch, left : left + patch]
        arrays.append(torch.from_numpy(crop.transpose(2, 0, 1).copy()).float() / 255.0)
    return (*arrays, torch.tensor([example["time"]], dtype=torch.float32))


def sample_batch(examples, batch_size: int, patch: int, rng: random.Random):
    values = [[], [], [], []]
    for _ in range(batch_size):
        example = rng.choice(examples)
        height, width = example["base"].shape[:2]
        top = rng.randrange(0, height - patch + 1)
        left = rng.randrange(0, width - patch + 1)
        for destination, tensor in zip(values, tensors_from_crop(example, top, left, patch)):
            destination.append(tensor)
    return tuple(torch.stack(items) for items in values)


def gradient_loss(prediction, target):
    px = prediction[:, :, :, 1:] - prediction[:, :, :, :-1]
    tx = target[:, :, :, 1:] - target[:, :, :, :-1]
    py = prediction[:, :, 1:, :] - prediction[:, :, :-1, :]
    ty = target[:, :, 1:, :] - target[:, :, :-1, :]
    return (px - tx).abs().mean() + (py - ty).abs().mean()


def write_video(cv2_module, frames, path: Path, fps: float):
    path.parent.mkdir(parents=True, exist_ok=True)
    height, width = frames[0].shape[:2]
    writer = cv2_module.VideoWriter(str(path), cv2_module.VideoWriter_fourcc(*"FFV1"), fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"cannot create {path}")
    for frame in frames:
        writer.write(cv2_module.cvtColor(frame, cv2_module.COLOR_RGB2BGR))
    writer.release()


@torch.inference_mode()
def infer_report(cv2_module, model, manifest: Path, base_csv: Path, candidate_csv: Path, out_dir: Path, device):
    base_report = read_report(base_csv)
    candidate_report = read_report(candidate_csv)
    report_rows = []
    rows = [row for row in read_jsonl(manifest) if row.get("view_type") == "wts_vehicle_or_ip_camera"]
    for position, row in enumerate(rows, start=1):
        sample_id = row["sample_id"]
        base_video = base_report.get(sample_id)
        candidate_video = candidate_report.get(sample_id)
        if base_video is None or candidate_video is None:
            raise RuntimeError(f"missing eval report row: {sample_id}")
        capture_base = cv2_module.VideoCapture(str(base_video))
        capture_candidate = cv2_module.VideoCapture(str(candidate_video))
        fps = float(capture_base.get(cv2_module.CAP_PROP_FPS) or 30.0)
        frame_count = int(capture_base.get(cv2_module.CAP_PROP_FRAME_COUNT))
        outputs = []
        for frame_index in range(frame_count):
            ok_base, base_bgr = capture_base.read()
            ok_candidate, candidate_bgr = capture_candidate.read()
            if not ok_base or not ok_candidate:
                break
            base = cv2_module.cvtColor(base_bgr, cv2_module.COLOR_BGR2RGB)
            candidate = cv2_module.cvtColor(candidate_bgr, cv2_module.COLOR_BGR2RGB)
            base_tensor = torch.from_numpy(base.transpose(2, 0, 1).copy()).float()[None].to(device) / 255.0
            candidate_tensor = torch.from_numpy(candidate.transpose(2, 0, 1).copy()).float()[None].to(device) / 255.0
            time_tensor = torch.tensor([[frame_index / float(max(1, frame_count - 1))]], device=device)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                multiplier = model(base_tensor, candidate_tensor, time_tensor)
                prediction = base_tensor + multiplier * (candidate_tensor - base_tensor)
            output = prediction[0].clamp(0, 1).float().cpu().permute(1, 2, 0).numpy()
            outputs.append((output * 255.0 + 0.5).astype(np.uint8))
        capture_base.release()
        capture_candidate.release()
        if len(outputs) != frame_count:
            raise RuntimeError(f"{sample_id}: decoded {len(outputs)}/{frame_count}")
        video_path = out_dir / "videos" / f"{sample_id}.avi"
        write_video(cv2_module, outputs, video_path, fps)
        report_rows.append({"sample_id": sample_id, "video": str(video_path), "status": "ok", "future_frames": frame_count - 1})
        print(f"eval [{position}/{len(rows)}] {sample_id} | ok", flush=True)
    with (out_dir / "report.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=report_rows[0].keys())
        writer.writeheader()
        writer.writerows(report_rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train_manifest", type=Path, required=True)
    parser.add_argument("--train_base_report", type=Path, required=True)
    parser.add_argument("--train_candidate_report", type=Path, required=True)
    parser.add_argument("--eval_manifest", type=Path, required=True)
    parser.add_argument("--eval_base_report", type=Path, required=True)
    parser.add_argument("--eval_candidate_report", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--train_limit", type=int, default=32)
    parser.add_argument("--train_width", type=int, default=640)
    parser.add_argument("--train_height", type=int, default=360)
    parser.add_argument("--patch", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--steps_per_epoch", type=int, default=100)
    parser.add_argument("--model_width", type=int, default=24)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--lambda_edge", type=float, default=0.10)
    parser.add_argument("--lambda_identity", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=20260621)
    parser.add_argument("--clean", action="store_true")
    args = parser.parse_args()

    cv2_module, _ = import_cv2_np()
    if args.clean and args.out_dir.exists():
        shutil.rmtree(args.out_dir)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "args.json").write_text(json.dumps(vars(args), default=str, indent=2), encoding="utf-8")
    examples = load_examples(
        cv2_module,
        args.train_manifest,
        args.train_base_report,
        args.train_candidate_report,
        args.train_limit,
        args.train_width,
        args.train_height,
    )
    device = torch.device("cuda")
    model = SupervisedDepthGate(args.model_width).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    rng = random.Random(args.seed)
    history = []
    model.train()
    for epoch in range(args.epochs):
        losses = []
        for _ in range(args.steps_per_epoch):
            base, candidate, target, time_value = sample_batch(examples, args.batch_size, args.patch, rng)
            base, candidate, target, time_value = [value.to(device) for value in (base, candidate, target, time_value)]
            multiplier = model(base, candidate, time_value)
            prediction = base + multiplier * (candidate - base)
            pixel = torch.sqrt((prediction - target) ** 2 + 1e-6).mean()
            edge = gradient_loss(prediction, target)
            identity = ((multiplier - 1.0) ** 2).mean()
            loss = pixel + args.lambda_edge * edge + args.lambda_identity * identity
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append((float(loss.item()), float(pixel.item()), float(edge.item()), float(identity.item())))
        means = np.mean(np.asarray(losses), axis=0)
        row = {"epoch": epoch + 1, "loss": means[0], "pixel": means[1], "edge": means[2], "identity": means[3]}
        history.append(row)
        print("epoch={epoch} loss={loss:.6f} pixel={pixel:.6f} edge={edge:.6f} identity={identity:.6f}".format(**row), flush=True)
    torch.save({"model": model.state_dict(), "args": vars(args), "examples": len(examples)}, args.out_dir / "supervised_depth_gate.pt")
    with (args.out_dir / "history.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=history[0].keys())
        writer.writeheader()
        writer.writerows(history)
    model.eval()
    infer_report(
        cv2_module,
        model,
        args.eval_manifest,
        args.eval_base_report,
        args.eval_candidate_report,
        args.out_dir,
        device,
    )
    print(f"examples={len(examples)}")
    print(f"output={args.out_dir}")


if __name__ == "__main__":
    main()

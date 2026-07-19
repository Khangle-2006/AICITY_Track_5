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

from baseline.frame_io import read_rgb
from wan_ticft.evaluation.evaluate_video_reports import (
    map_target_offsets,
    read_report,
    read_video_rgb_frames,
    video_fps,
)


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def resize_rgb(image: np.ndarray, width: int, height: int) -> np.ndarray:
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)


def load_examples(
    manifest: Path,
    base_csv: Path,
    candidate_csv: Path,
    view_type: str,
    limit: int,
    width: int,
    height: int,
) -> list[dict[str, object]]:
    base_report = read_report(base_csv)
    candidate_report = read_report(candidate_csv)
    examples: list[dict[str, object]] = []
    rows = [row for row in read_jsonl(manifest) if row.get("view_type") == view_type]
    accepted = 0
    for row in rows:
        if accepted >= limit:
            break
        sample_id = row["sample_id"]
        base_video = base_report.get(sample_id)
        candidate_video = candidate_report.get(sample_id)
        if base_video is None or candidate_video is None:
            continue
        offsets = map_target_offsets(row, video_fps(cv2, base_video))
        base_map = read_video_rgb_frames(cv2, base_video, offsets)
        candidate_map = read_video_rgb_frames(cv2, candidate_video, offsets)
        targets = [read_rgb(cv2, Path(path)) for path in (row.get("frames") or {}).get("target", [])]
        if len(targets) != len(offsets) or any(target is None for target in targets):
            continue
        max_offset = max(offsets) if offsets else 1
        added = 0
        for offset, target in zip(offsets, targets):
            if offset not in base_map or offset not in candidate_map or target is None:
                continue
            examples.append(
                {
                    "sample_id": sample_id,
                    "base": resize_rgb(base_map[offset], width, height),
                    "candidate": resize_rgb(candidate_map[offset], width, height),
                    "target": resize_rgb(target, width, height),
                    "time": offset / float(max(1, max_offset)),
                }
            )
            added += 1
        if added:
            accepted += 1
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

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return F.silu(value + self.body(value))


class ResidualGate(nn.Module):
    def __init__(self, width: int, blocks: int, max_delta: float):
        super().__init__()
        self.max_delta = max_delta
        self.stem = nn.Sequential(nn.Conv2d(9, width, 3, padding=1), nn.SiLU())
        self.blocks = nn.Sequential(*[ResidualBlock(width) for _ in range(blocks)])
        self.head = nn.Conv2d(width, 1, 3, padding=1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, base: torch.Tensor, candidate: torch.Tensor, time_value: torch.Tensor) -> torch.Tensor:
        residual = candidate - base
        magnitude = residual.abs().mean(dim=1, keepdim=True)
        base_luma = base.mean(dim=1, keepdim=True)
        time_plane = time_value[:, :, None, None].expand(-1, -1, base.shape[2], base.shape[3])
        features = torch.cat([base, residual, magnitude, base_luma, time_plane], dim=1)
        raw = self.head(self.blocks(self.stem(features)))
        return 1.0 + self.max_delta * torch.tanh(raw)


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


def gradient_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    px = prediction[:, :, :, 1:] - prediction[:, :, :, :-1]
    tx = target[:, :, :, 1:] - target[:, :, :, :-1]
    py = prediction[:, :, 1:, :] - prediction[:, :, :-1, :]
    ty = target[:, :, 1:, :] - target[:, :, :-1, :]
    return (px - tx).abs().mean() + (py - ty).abs().mean()


def write_video(frames: list[np.ndarray], path: Path, fps: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    height, width = frames[0].shape[:2]
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"FFV1"), fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"cannot create {path}")
    for frame in frames:
        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    writer.release()


@torch.inference_mode()
def infer_report(
    model: ResidualGate,
    manifest: Path,
    base_csv: Path,
    candidate_csv: Path,
    view_type: str,
    out_dir: Path,
    device: torch.device,
) -> None:
    base_report = read_report(base_csv)
    candidate_report = read_report(candidate_csv)
    rows = [row for row in read_jsonl(manifest) if row.get("view_type") == view_type]
    report_rows = []
    for position, row in enumerate(rows, start=1):
        sample_id = row["sample_id"]
        base_video = base_report.get(sample_id)
        candidate_video = candidate_report.get(sample_id)
        if base_video is None or candidate_video is None:
            raise RuntimeError(f"{sample_id}: missing base/candidate report row")
        cap_base = cv2.VideoCapture(str(base_video))
        cap_candidate = cv2.VideoCapture(str(candidate_video))
        fps = float(cap_base.get(cv2.CAP_PROP_FPS) or 30.0)
        frame_count = min(
            int(cap_base.get(cv2.CAP_PROP_FRAME_COUNT) or 0),
            int(cap_candidate.get(cv2.CAP_PROP_FRAME_COUNT) or 0),
        )
        outputs: list[np.ndarray] = []
        for frame_index in range(frame_count):
            ok_base, base_bgr = cap_base.read()
            ok_candidate, candidate_bgr = cap_candidate.read()
            if not ok_base or not ok_candidate:
                break
            base = cv2.cvtColor(base_bgr, cv2.COLOR_BGR2RGB)
            candidate = cv2.cvtColor(candidate_bgr, cv2.COLOR_BGR2RGB)
            if base.shape[:2] != candidate.shape[:2]:
                candidate = cv2.resize(candidate, (base.shape[1], base.shape[0]), interpolation=cv2.INTER_AREA)
            base_tensor = torch.from_numpy(base.transpose(2, 0, 1).copy()).float()[None].to(device) / 255.0
            cand_tensor = torch.from_numpy(candidate.transpose(2, 0, 1).copy()).float()[None].to(device) / 255.0
            time_tensor = torch.tensor([[frame_index / float(max(1, frame_count - 1))]], device=device)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                multiplier = model(base_tensor, cand_tensor, time_tensor)
                prediction = base_tensor + multiplier * (cand_tensor - base_tensor)
            image = prediction[0].clamp(0, 1).float().cpu().permute(1, 2, 0).numpy()
            outputs.append((image * 255.0 + 0.5).astype(np.uint8))
        cap_base.release()
        cap_candidate.release()
        if len(outputs) != frame_count or not outputs:
            raise RuntimeError(f"{sample_id}: decoded {len(outputs)}/{frame_count}")
        video_path = out_dir / "videos" / f"{sample_id}.avi"
        write_video(outputs, video_path, fps)
        report_rows.append({"sample_id": sample_id, "video": str(video_path), "status": "ok", "future_frames": len(outputs)})
        print(f"eval [{position}/{len(rows)}] {sample_id} | frames={len(outputs)}", flush=True)
    with (out_dir / "report.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=report_rows[0].keys())
        writer.writeheader()
        writer.writerows(report_rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a supervised residual multiplier over a frozen base/candidate pair.")
    parser.add_argument("--train_manifest", type=Path, required=True)
    parser.add_argument("--train_base_report", type=Path, required=True)
    parser.add_argument("--train_candidate_report", type=Path, required=True)
    parser.add_argument("--eval_manifest", type=Path, required=True)
    parser.add_argument("--eval_base_report", type=Path, required=True)
    parser.add_argument("--eval_candidate_report", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--view_type", default="bdd")
    parser.add_argument("--train_limit", type=int, default=64)
    parser.add_argument("--train_width", type=int, default=640)
    parser.add_argument("--train_height", type=int, default=360)
    parser.add_argument("--patch", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--steps_per_epoch", type=int, default=150)
    parser.add_argument("--model_width", type=int, default=32)
    parser.add_argument("--model_blocks", type=int, default=5)
    parser.add_argument("--max_delta", type=float, default=0.75)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--lambda_edge", type=float, default=0.10)
    parser.add_argument("--lambda_identity", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=20260623)
    parser.add_argument("--clean", action="store_true")
    args = parser.parse_args()

    if args.clean and args.out_dir.exists():
        shutil.rmtree(args.out_dir)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "args.json").write_text(json.dumps(vars(args), default=str, indent=2), encoding="utf-8")

    examples = load_examples(
        args.train_manifest,
        args.train_base_report,
        args.train_candidate_report,
        args.view_type,
        args.train_limit,
        args.train_width,
        args.train_height,
    )
    device = torch.device("cuda")
    model = ResidualGate(args.model_width, args.model_blocks, args.max_delta).to(device)
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
        print(
            "epoch={epoch} loss={loss:.6f} pixel={pixel:.6f} edge={edge:.6f} identity={identity:.6f}".format(
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
    infer_report(model, args.eval_manifest, args.eval_base_report, args.eval_candidate_report, args.view_type, args.out_dir, device)
    print(f"examples={len(examples)}")
    print(f"output={args.out_dir}")


if __name__ == "__main__":
    main()

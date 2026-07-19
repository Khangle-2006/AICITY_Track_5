"""Train a conservative temporal deformation refiner on matched base predictions.

The network cannot synthesize RGB. It predicts only a small, smooth correction
field that warps the original base pixels, preserving texture and appearance.
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
from wan_ticft.evaluation.evaluate_video_reports import (
    map_target_offsets,
    read_report,
    read_video_rgb_frames,
    video_fps,
)


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def resize(cv2, image, width: int, height: int):
    if image.shape[:2] == (height, width):
        return image
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)


def load_images(cv2, paths: list[str]):
    return [image for path in paths if (image := read_rgb(cv2, Path(path))) is not None]


def load_base_frames(cv2, row: dict, video: Path):
    offsets = map_target_offsets(row, video_fps(cv2, video))
    frame_map = read_video_rgb_frames(cv2, video, offsets)
    return [frame_map[index] for index in offsets if index in frame_map]


def dense_flow(cv2, previous, current):
    first = cv2.cvtColor(previous, cv2.COLOR_RGB2GRAY)
    second = cv2.cvtColor(current, cv2.COLOR_RGB2GRAY)
    return cv2.calcOpticalFlowFarneback(first, second, None, 0.5, 4, 21, 4, 7, 1.5, 0)


def prepare_sequence(cv2, row: dict, video: Path, width: int, height: int):
    bases = load_base_frames(cv2, row, video)
    targets = load_images(cv2, (row.get("frames") or {}).get("target", []))
    conditions = load_images(cv2, (row.get("frames") or {}).get("condition", []))
    if len(bases) != len(targets) or len(bases) < 8 or not conditions:
        return None
    bases = np.stack([resize(cv2, frame, width, height) for frame in bases])
    targets = np.stack([resize(cv2, frame, width, height) for frame in targets])
    anchor = resize(cv2, conditions[-1], width, height)
    flows = []
    previous = anchor
    for frame in bases:
        flows.append(dense_flow(cv2, previous, frame))
        previous = frame
    return {
        "sample_id": row["sample_id"],
        "base": bases,
        "target": targets,
        "anchor": anchor,
        "flow": np.stack(flows).astype(np.float16),
    }


def build_model(nn, torch, width: int, max_flow_px: float):
    class Block(nn.Module):
        def __init__(self, channels: int):
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv3d(channels, channels, 3, padding=1),
                nn.GroupNorm(8, channels),
                nn.SiLU(),
                nn.Conv3d(channels, channels, 3, padding=1),
                nn.GroupNorm(8, channels),
            )

        def forward(self, value):
            return torch.nn.functional.silu(value + self.net(value))

    class DeformationRefiner(nn.Module):
        def __init__(self):
            super().__init__()
            self.stem = nn.Sequential(nn.Conv3d(12, width, 3, padding=1), nn.SiLU())
            self.body = nn.Sequential(*[Block(width) for _ in range(5)])
            self.head = nn.Conv3d(width, 2, 3, padding=1)
            nn.init.zeros_(self.head.weight)
            nn.init.zeros_(self.head.bias)

        def forward(self, value):
            return torch.tanh(self.head(self.body(self.stem(value)))) * max_flow_px

    return DeformationRefiner()


def make_input(torch, sequence: dict, start: int, length: int, device):
    base = torch.from_numpy(sequence["base"][start : start + length]).to(device).float().div(255)
    target = torch.from_numpy(sequence["target"][start : start + length]).to(device).float().div(255)
    anchor = torch.from_numpy(sequence["anchor"]).to(device).float().div(255)
    anchor = anchor.unsqueeze(0).expand(length, -1, -1, -1)
    if start == 0:
        previous = torch.cat([anchor[:1], base[:-1]], dim=0)
    else:
        previous = torch.from_numpy(sequence["base"][start - 1 : start + length - 1]).to(device).float().div(255)
    delta = base - previous
    flow = torch.from_numpy(sequence["flow"][start : start + length]).to(device).float()
    height, width = base.shape[1:3]
    flow[..., 0] /= max(width, 1)
    flow[..., 1] /= max(height, 1)
    time = torch.linspace(start, start + length - 1, length, device=device)
    time = (time / max(1, len(sequence["base"]) - 1)).view(length, 1, 1, 1).expand(-1, height, width, -1)
    features = torch.cat([base, anchor, delta, flow, time], dim=3)
    features = features.permute(3, 0, 1, 2).unsqueeze(0).contiguous()
    base = base.permute(0, 3, 1, 2).unsqueeze(0).contiguous()
    target = target.permute(0, 3, 1, 2).unsqueeze(0).contiguous()
    return features, base, target


def warp_torch(torch, F, frames, flow):
    batch, time, _, height, width = frames.shape
    y, x = torch.meshgrid(
        torch.linspace(-1, 1, height, device=frames.device),
        torch.linspace(-1, 1, width, device=frames.device),
        indexing="ij",
    )
    grid = torch.stack([x, y], dim=-1).view(1, 1, height, width, 2).expand(batch, time, -1, -1, -1).clone()
    correction = flow.permute(0, 2, 3, 4, 1)
    grid[..., 0] += correction[..., 0] * (2.0 / max(1, width - 1))
    grid[..., 1] += correction[..., 1] * (2.0 / max(1, height - 1))
    values = frames.reshape(batch * time, 3, height, width)
    grids = grid.reshape(batch * time, height, width, 2)
    return F.grid_sample(values, grids, mode="bilinear", padding_mode="border", align_corners=True).view(
        batch, time, 3, height, width
    )


def ssim_loss(F, prediction, target):
    pred = prediction.flatten(0, 1)
    truth = target.flatten(0, 1)
    mu_p = F.avg_pool2d(pred, 7, 1, 3)
    mu_t = F.avg_pool2d(truth, 7, 1, 3)
    var_p = F.avg_pool2d(pred * pred, 7, 1, 3) - mu_p * mu_p
    var_t = F.avg_pool2d(truth * truth, 7, 1, 3) - mu_t * mu_t
    cov = F.avg_pool2d(pred * truth, 7, 1, 3) - mu_p * mu_t
    score = ((2 * mu_p * mu_t + 0.01**2) * (2 * cov + 0.03**2)) / (
        (mu_p * mu_p + mu_t * mu_t + 0.01**2) * (var_p + var_t + 0.03**2) + 1e-8
    )
    return 1.0 - score.mean()


def video_feature_loss(torch, F, model, prediction, target):
    mean = torch.tensor((0.43216, 0.394666, 0.37645), device=prediction.device)[None, :, None, None, None]
    std = torch.tensor((0.22803, 0.22145, 0.216989), device=prediction.device)[None, :, None, None, None]

    def encode(value):
        value = value.permute(0, 2, 1, 3, 4)
        batch, channels, time, height, width = value.shape
        value = F.interpolate(
            value.permute(0, 2, 1, 3, 4).reshape(batch * time, channels, height, width),
            size=(112, 112),
            mode="bilinear",
            align_corners=False,
        ).view(batch, time, channels, 112, 112).permute(0, 2, 1, 3, 4)
        return F.normalize(model((value - mean) / std), dim=1)

    predicted_feature = encode(prediction)
    with torch.no_grad():
        target_feature = encode(target)
    return (predicted_feature - target_feature).abs().mean()


def resize_flow(cv2, flow, width: int, height: int):
    source_height, source_width = flow.shape[:2]
    output = cv2.resize(flow, (width, height), interpolation=cv2.INTER_LINEAR)
    output[..., 0] *= width / float(source_width)
    output[..., 1] *= height / float(source_height)
    return output


def warp_cv2(cv2, frame, flow):
    height, width = frame.shape[:2]
    x, y = np.meshgrid(np.arange(width, dtype=np.float32), np.arange(height, dtype=np.float32))
    return cv2.remap(
        frame,
        x + flow[..., 0],
        y + flow[..., 1],
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )


def predict_flow(torch, model, sequence: dict, window: int, device):
    del window
    features, _, _ = make_input(torch, sequence, 0, len(sequence["base"]), device)
    with torch.no_grad():
        return model(features)[0].permute(1, 2, 3, 0).float().cpu().numpy()


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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train_manifest", type=Path, required=True)
    parser.add_argument("--train_report", type=Path, required=True)
    parser.add_argument("--eval_manifest", type=Path, required=True)
    parser.add_argument("--eval_report", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--train_limit", type=int, default=16)
    parser.add_argument("--eval_limit", type=int, default=24)
    parser.add_argument("--width", type=int, default=320)
    parser.add_argument("--height", type=int, default=180)
    parser.add_argument("--window", type=int, default=8)
    parser.add_argument("--steps", type=int, default=800)
    parser.add_argument("--model_width", type=int, default=32)
    parser.add_argument("--max_flow_px", type=float, default=3.0)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--lambda_ssim", type=float, default=0.15)
    parser.add_argument("--lambda_temporal", type=float, default=0.20)
    parser.add_argument("--lambda_video", type=float, default=0.04)
    parser.add_argument("--lambda_magnitude", type=float, default=0.01)
    parser.add_argument("--lambda_smooth", type=float, default=0.03)
    parser.add_argument("--alphas", type=float, nargs="+", default=(0.25, 0.5, 0.75, 1.0))
    parser.add_argument("--seed", type=int, default=20260622)
    parser.add_argument("--clean", action="store_true")
    args = parser.parse_args()

    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torchvision.models.video import R3D_18_Weights, r3d_18

    cv2, _ = import_cv2_np()
    if args.clean and args.out_dir.exists():
        shutil.rmtree(args.out_dir)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "args.json").write_text(json.dumps(vars(args), default=str, indent=2), encoding="utf-8")

    train_rows = [row for row in read_jsonl(args.train_manifest) if row.get("view_type") == "bdd"][: args.train_limit]
    train_report = read_report(args.train_report)
    sequences = []
    for position, row in enumerate(train_rows, start=1):
        video = train_report.get(row["sample_id"])
        sequence = prepare_sequence(cv2, row, video, args.width, args.height) if video else None
        if sequence is not None:
            sequences.append(sequence)
        print(f"prepare train [{position}/{len(train_rows)}] {row['sample_id']} | {'ok' if sequence else 'skip'}", flush=True)
    if len(sequences) != args.train_limit:
        raise RuntimeError(f"expected {args.train_limit} train sequences, found {len(sequences)}")

    device = torch.device("cuda")
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    model = build_model(nn, torch, args.model_width, args.max_flow_px).to(device)
    feature_model = r3d_18(weights=R3D_18_Weights.DEFAULT)
    feature_model.fc = nn.Identity()
    feature_model.eval().to(device)
    for parameter in feature_model.parameters():
        parameter.requires_grad_(False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    history = []
    model.train()
    for step in range(1, args.steps + 1):
        sequence = rng.choice(sequences)
        start = rng.randrange(0, len(sequence["base"]) - args.window + 1)
        features, base, target = make_input(torch, sequence, start, args.window, device)
        flow = model(features)
        prediction = warp_torch(torch, F, base, flow)
        charbonnier = ((prediction - target).square() + 1e-6).sqrt().mean()
        structural = ssim_loss(F, prediction, target)
        temporal = ((prediction[:, 1:] - prediction[:, :-1]) - (target[:, 1:] - target[:, :-1])).abs().mean()
        feature = video_feature_loss(torch, F, feature_model, prediction, target)
        magnitude = flow.abs().mean()
        spatial = (flow[..., 1:] - flow[..., :-1]).abs().mean() + (flow[..., 1:, :] - flow[..., :-1, :]).abs().mean()
        acceleration = (flow[:, :, 2:] - 2 * flow[:, :, 1:-1] + flow[:, :, :-2]).abs().mean()
        smooth = spatial + acceleration
        loss = (
            charbonnier
            + args.lambda_ssim * structural
            + args.lambda_temporal * temporal
            + args.lambda_video * feature
            + args.lambda_magnitude * magnitude
            + args.lambda_smooth * smooth
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step == 1 or step % 50 == 0:
            row = {
                "step": step,
                "loss": float(loss.detach().cpu()),
                "pixel": float(charbonnier.detach().cpu()),
                "ssim": float(structural.detach().cpu()),
                "temporal": float(temporal.detach().cpu()),
                "video": float(feature.detach().cpu()),
                "flow_px": float(magnitude.detach().cpu()),
                "smooth": float(smooth.detach().cpu()),
            }
            history.append(row)
            print(" ".join(f"{key}={value:.6f}" if key != "step" else f"step={value}" for key, value in row.items()), flush=True)

    torch.save({"model": model.state_dict(), "args": vars(args)}, args.out_dir / "base_deformation_refiner.pt")
    write_csv = args.out_dir / "history.csv"
    with write_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=history[0])
        writer.writeheader()
        writer.writerows(history)

    eval_rows = [row for row in read_jsonl(args.eval_manifest) if row.get("view_type") == "bdd"][: args.eval_limit]
    eval_report = read_report(args.eval_report)
    reports = {alpha: [] for alpha in args.alphas}
    diagnostics = []
    model.eval()
    for position, row in enumerate(eval_rows, start=1):
        video = eval_report.get(row["sample_id"])
        sequence = prepare_sequence(cv2, row, video, args.width, args.height) if video else None
        if sequence is None:
            raise RuntimeError(f"cannot prepare eval sequence: {row['sample_id']}")
        correction = predict_flow(torch, model, sequence, args.window, device)
        full_bases = load_base_frames(cv2, row, video)
        fps = video_fps(cv2, video) or 30.0
        for alpha in args.alphas:
            frames = []
            for frame, low_flow in zip(full_bases, correction):
                full_flow = resize_flow(cv2, low_flow * alpha, frame.shape[1], frame.shape[0])
                frames.append(warp_cv2(cv2, frame, full_flow))
            tag = f"a{int(round(alpha * 100)):03d}"
            output = args.out_dir / f"videos_{tag}" / f"{row['sample_id']}.avi"
            write_video(cv2, frames, output, fps)
            reports[alpha].append({"sample_id": row["sample_id"], "view_type": "bdd", "video": str(output), "status": "ok"})
        diagnostics.append(
            {
                "sample_id": row["sample_id"],
                "mean_flow_px": float(np.mean(np.linalg.norm(correction, axis=3))),
                "p95_flow_px": float(np.percentile(np.linalg.norm(correction, axis=3), 95)),
                "max_flow_px": float(np.max(np.linalg.norm(correction, axis=3))),
            }
        )
        print(f"eval [{position}/{len(eval_rows)}] {row['sample_id']} | ok", flush=True)

    for alpha, rows in reports.items():
        tag = f"a{int(round(alpha * 100)):03d}"
        with (args.out_dir / f"report_{tag}.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=rows[0])
            writer.writeheader()
            writer.writerows(rows)
    with (args.out_dir / "diagnostics.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=diagnostics[0])
        writer.writeheader()
        writer.writerows(diagnostics)


if __name__ == "__main__":
    main()

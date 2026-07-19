"""Train a motion-only future predictor: learned flow + confidence, no RGB synth."""

from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
from pathlib import Path

from baseline.frame_io import import_cv2_np, read_rgb
from baseline.metrics import mean_absdiff, psnr, ssim_gray
from baseline.refiner_dataset import filter_rows, load_split_rows, sample_rows
from baseline.video_io import extract_val_history_and_target
from wan_ticft.evaluation.train_metric_residual_adapter import ssim_loss


def read_jsonl(path: Path) -> list[dict]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def read_target_counts(path: Path | None) -> dict[str, int]:
    if not path:
        return {}
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        out = {}
        for row in csv.DictReader(handle):
            try:
                out[row["sample_id"]] = int(row.get("target_N") or row.get("future_frames") or 0)
            except (KeyError, TypeError, ValueError):
                continue
        return out


def load_images(cv2, paths):
    return [image for path in paths if (image := read_rgb(cv2, Path(path))) is not None]


def resize(cv2, image, width: int, height: int):
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)


def make_input(np, cv2, frames, horizon: int, phase: str, width: int, height: int, max_horizon: int):
    values = [resize(cv2, frame, width, height).astype(np.float32) / 255.0 for frame in frames[-4:]]
    while len(values) < 4:
        values.insert(0, values[0])
    delta = np.mean(np.abs(values[-1] - values[-2]), axis=2, keepdims=True)
    context = np.zeros((height, width, 6), dtype=np.float32)
    context[:, :, 0] = min(1.0, horizon / float(max_horizon))
    try:
        phase_index = int(phase)
    except (TypeError, ValueError):
        phase_index = -1
    if 0 <= phase_index < 5:
        context[:, :, 1 + phase_index] = 1.0
    return np.concatenate([*values, delta, context], axis=2)


def to_tensor(torch, np, image):
    return torch.from_numpy(image.astype(np.float32) / 255.0).permute(2, 0, 1).float()


class FutureDataset:
    def __init__(self, cv2, np, torch, rows, args):
        self.cv2, self.np, self.torch = cv2, np, torch
        self.rows, self.args = rows, args
        self.rng = random.Random(args.seed + 31)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        try:
            row = self.rows[index]
            horizon = self.rng.randint(1, self.args.max_horizon)
            frames, target, meta = extract_val_history_and_target(self.cv2, row, 4, horizon)
            if len(frames) < 2 or target is None or (meta and "error" in meta):
                return None
            x = make_input(
                self.np, self.cv2, frames, horizon, row.get("phase", ""),
                self.args.width, self.args.height, self.args.max_horizon,
            )
            y = resize(self.cv2, target, self.args.width, self.args.height)
            return self.torch.from_numpy(x).permute(2, 0, 1).float(), to_tensor(self.torch, self.np, y)
        except (RuntimeError, OSError, ValueError):
            return None


def collate(batch):
    valid = [item for item in batch if item is not None]
    if not valid:
        return None
    import torch

    return torch.stack([item[0] for item in valid]), torch.stack([item[1] for item in valid])


def warp(torch, F, image, flow):
    batch, _, height, width = image.shape
    yy, xx = torch.meshgrid(
        torch.linspace(-1.0, 1.0, height, device=image.device, dtype=image.dtype),
        torch.linspace(-1.0, 1.0, width, device=image.device, dtype=image.dtype),
        indexing="ij",
    )
    base = torch.stack([xx, yy], dim=-1).unsqueeze(0).repeat(batch, 1, 1, 1)
    fx = flow[:, 0] * (2.0 / max(1, width - 1))
    fy = flow[:, 1] * (2.0 / max(1, height - 1))
    return F.grid_sample(image, base + torch.stack([fx, fy], dim=-1), mode="bilinear", padding_mode="border", align_corners=True)


def build_model(torch, nn, F, in_channels: int, width: int, max_flow: float, alpha_bias: float):
    class Block(nn.Module):
        def __init__(self, channels):
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv2d(channels, channels, 3, padding=1),
                nn.GroupNorm(8, channels),
                nn.SiLU(inplace=True),
                nn.Conv2d(channels, channels, 3, padding=1),
                nn.GroupNorm(8, channels),
            )

        def forward(self, x):
            return F.silu(x + self.net(x))

    class FlowConfidence(nn.Module):
        def __init__(self):
            super().__init__()
            self.stem = nn.Sequential(nn.Conv2d(in_channels, width, 5, padding=2), nn.SiLU(inplace=True))
            self.down1 = nn.Sequential(nn.Conv2d(width, width * 2, 4, stride=2, padding=1), nn.SiLU(inplace=True), Block(width * 2))
            self.down2 = nn.Sequential(nn.Conv2d(width * 2, width * 4, 4, stride=2, padding=1), nn.SiLU(inplace=True), Block(width * 4))
            self.mid = nn.Sequential(Block(width * 4), Block(width * 4), Block(width * 4))
            self.up1 = nn.Sequential(nn.Conv2d(width * 6, width * 2, 3, padding=1), nn.SiLU(inplace=True), Block(width * 2))
            self.up2 = nn.Sequential(nn.Conv2d(width * 3, width, 3, padding=1), nn.SiLU(inplace=True), Block(width))
            self.head = nn.Conv2d(width, 3, 3, padding=1)

        def forward(self, x):
            last = x[:, 9:12]
            s0 = self.stem(x)
            s1 = self.down1(s0)
            s2 = self.down2(s1)
            y = self.mid(s2)
            y = F.interpolate(y, size=s1.shape[-2:], mode="bilinear", align_corners=False)
            y = self.up1(torch.cat([y, s1], dim=1))
            y = F.interpolate(y, size=s0.shape[-2:], mode="bilinear", align_corners=False)
            y = self.up2(torch.cat([y, s0], dim=1))
            raw = self.head(y)
            flow = torch.tanh(raw[:, :2]) * max_flow
            alpha = torch.sigmoid(raw[:, 2:3] + alpha_bias)
            warped = warp(torch, F, last, flow)
            return ((1.0 - alpha) * last + alpha * warped).clamp(0.0, 1.0), warped, flow, alpha

    return FlowConfidence()


def edge_loss(prediction, target):
    dx_p = prediction[:, :, :, 1:] - prediction[:, :, :, :-1]
    dx_t = target[:, :, :, 1:] - target[:, :, :, :-1]
    dy_p = prediction[:, :, 1:, :] - prediction[:, :, :-1, :]
    dy_t = target[:, :, 1:, :] - target[:, :, :-1, :]
    return (dx_p - dx_t).abs().mean() + (dy_p - dy_t).abs().mean()


def smooth_loss(flow):
    return (flow[:, :, :, 1:] - flow[:, :, :, :-1]).abs().mean() + (flow[:, :, 1:, :] - flow[:, :, :-1, :]).abs().mean()


def train_model(torch, nn, F, cv2, np, args, device):
    rows = filter_rows(load_split_rows(Path(args.index_dir), "train"), view_type=args.view_type)
    rows = sample_rows(rows, args.max_train_rows, args.seed)
    loader = torch.utils.data.DataLoader(
        FutureDataset(cv2, np, torch, rows, args),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate,
    )
    model = build_model(torch, nn, F, 19, args.model_width, args.max_flow, args.alpha_bias).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        totals = {"loss": 0.0, "l1": 0.0, "ssim": 0.0, "edge": 0.0, "flow": 0.0, "alpha": 0.0, "alpha_mean": 0.0}
        count = 0
        for batch in loader:
            if batch is None:
                continue
            x, y = [value.to(device) for value in batch]
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=args.amp):
                pred, _, flow, alpha = model(x)
                l1 = (pred - y).abs().mean()
                sl = ssim_loss(torch, F, pred, y)
                el = edge_loss(pred, y)
                fl = smooth_loss(flow) / max(args.max_flow, 1.0)
                al = alpha.mean()
                loss = l1 + args.lambda_ssim * sl + args.lambda_edge * el + args.lambda_flow * fl + args.lambda_alpha * al
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            batch_n = x.shape[0]
            count += batch_n
            for name, value in (("loss", loss), ("l1", l1), ("ssim", sl), ("edge", el), ("flow", fl), ("alpha", al), ("alpha_mean", al)):
                totals[name] += float(value.detach().cpu()) * batch_n
        row = {"epoch": epoch, "samples": count, **{key: value / max(1, count) for key, value in totals.items()}}
        history.append(row)
        print(" ".join(f"{key}={value:.6f}" if isinstance(value, float) else f"{key}={value}" for key, value in row.items()), flush=True)
    return model, len(rows), history


def score(cv2, np, predictions, targets):
    values = [(psnr(cv2, np, p, t), ssim_gray(cv2, np, p, t), mean_absdiff(np, p, t)) for p, t in zip(predictions, targets)]
    return tuple(float(np.mean([value[index] for value in values])) for index in range(3))


def write_video(cv2, frames, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    height, width = frames[0].shape[:2]
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"FFV1"), 30.0, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"cannot create {path}")
    for frame in frames:
        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    writer.release()


def evaluate(torch, cv2, np, model, args, device):
    rows_out, report_rows = [], []
    target_counts = read_target_counts(args.target_count_report)
    model.eval()
    for position, row in enumerate(read_jsonl(Path(args.manifest)), start=1):
        if row.get("view_type") != args.view_type:
            continue
        paths = row.get("frames") or {}
        conditions = load_images(cv2, paths.get("condition") or [])
        targets = load_images(cv2, paths.get("target") or [])
        if len(conditions) < 2:
            continue
        indices = row.get("frame_indices") or {}
        if targets and indices.get("target"):
            last_condition = int(indices["condition"][-1])
            offsets = [max(1, int(index) - last_condition) for index in indices["target"]]
            references = targets
        else:
            target_count = int(target_counts.get(row["sample_id"]) or (row.get("ticft") or {}).get("target_frames") or args.target_frames)
            offsets = list(range(1, target_count + 1))
            references = [conditions[-1]] * target_count
        predictions, alpha_means = [], []
        alpha_dir = args.out_dir / "alpha" / row["sample_id"]
        alpha_dir.mkdir(parents=True, exist_ok=True)
        for horizon, target in zip(offsets, references):
            x = make_input(np, cv2, conditions, horizon, row.get("phase", ""), args.width, args.height, args.max_horizon)
            tensor = torch.from_numpy(x).permute(2, 0, 1).unsqueeze(0).float().to(device)
            with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=args.amp):
                pred, _, _, alpha = model(tensor)
            image = pred[0].float().cpu().clamp(0, 1).permute(1, 2, 0).numpy()
            image = (image * 255.0 + 0.5).astype(np.uint8)
            if image.shape[:2] != target.shape[:2]:
                image = cv2.resize(image, (target.shape[1], target.shape[0]), interpolation=cv2.INTER_CUBIC)
            predictions.append(image)
            alpha_np = alpha[0, 0].float().cpu().numpy()
            if alpha_np.shape[:2] != target.shape[:2]:
                alpha_np = cv2.resize(alpha_np, (target.shape[1], target.shape[0]), interpolation=cv2.INTER_LINEAR)
            cv2.imwrite(str(alpha_dir / f"alpha_{len(predictions)-1:03d}.png"), np.clip(alpha_np * 255.0, 0, 255).astype(np.uint8))
            alpha_means.append(float(alpha_np.mean()))
        if targets:
            repeat = [conditions[-1]] * len(targets)
            for candidate, frames in (("repeat_last", repeat), ("flow_conf", predictions)):
                p, s, d = score(cv2, np, frames, targets)
                rows_out.append({"sample_id": row["sample_id"], "phase": row.get("phase", ""), "candidate": candidate, "psnr": f"{p:.6f}", "ssim": f"{s:.6f}", "absdiff": f"{d:.6f}", "alpha_mean": f"{np.mean(alpha_means):.6f}" if alpha_means else ""})
        video = args.out_dir / "videos" / f"{row['sample_id']}.avi"
        write_video(cv2, predictions, video)
        report_rows.append({"sample_id": row["sample_id"], "video": str(video), "status": "ok", "future_frames": len(predictions), "alpha_mean": f"{np.mean(alpha_means):.6f}"})
        print(f"eval [{position}] {row['sample_id']} | ok alpha={np.mean(alpha_means):.4f}", flush=True)
    return rows_out, report_rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index_dir", default="/workspace/experiments/index")
    parser.add_argument("--manifest", default="/workspace/experiments/wan_ticft/manifests/val_ticft_balanced_125_frames.jsonl")
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--max_train_rows", type=int, default=1000)
    parser.add_argument("--width", type=int, default=320)
    parser.add_argument("--height", type=int, default=180)
    parser.add_argument("--max_horizon", type=int, default=90)
    parser.add_argument("--target_frames", type=int, default=9)
    parser.add_argument("--target_count_report", type=Path, default=None)
    parser.add_argument("--view_type", default="bdd")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=6)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--model_width", type=int, default=48)
    parser.add_argument("--max_flow", type=float, default=48.0)
    parser.add_argument("--alpha_bias", type=float, default=-1.4)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--lambda_ssim", type=float, default=0.20)
    parser.add_argument("--lambda_edge", type=float, default=0.10)
    parser.add_argument("--lambda_flow", type=float, default=0.01)
    parser.add_argument("--lambda_alpha", type=float, default=0.005)
    parser.add_argument("--seed", type=int, default=20260701)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--clean", action="store_true")
    args = parser.parse_args()

    if args.clean and args.out_dir.exists():
        shutil.rmtree(args.out_dir)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "args.json").write_text(json.dumps(vars(args), default=str, indent=2), encoding="utf-8")

    cv2, np = import_cv2_np()
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device} size={args.width}x{args.height} rows={args.max_train_rows} epochs={args.epochs}", flush=True)
    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint, map_location=device)
        saved = checkpoint.get("args", {})
        args.model_width = int(saved.get("model_width", args.model_width))
        args.max_flow = float(saved.get("max_flow", args.max_flow))
        args.alpha_bias = float(saved.get("alpha_bias", args.alpha_bias))
        model = build_model(torch, nn, F, 19, args.model_width, args.max_flow, args.alpha_bias).to(device)
        model.load_state_dict(checkpoint["model"])
        print(f"loaded checkpoint {args.checkpoint}", flush=True)
    else:
        model, train_rows, history = train_model(torch, nn, F, cv2, np, args, device)
        torch.save({"model": model.state_dict(), "args": vars(args), "train_rows": train_rows}, args.out_dir / "flow_conf.pt")
        with (args.out_dir / "history.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=history[0].keys())
            writer.writeheader()
            writer.writerows(history)
    rows, report_rows = evaluate(torch, cv2, np, model, args, device)
    if rows:
        with (args.out_dir / "eval.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
    with (args.out_dir / "report.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=report_rows[0].keys())
        writer.writeheader()
        writer.writerows(report_rows)
    if rows:
        print()
        print(f"{'Candidate':<14} {'N':>5} {'PSNR':>10} {'SSIM':>10} {'AbsDiff':>10}")
        print("-" * 54)
        for candidate in ("repeat_last", "flow_conf"):
            items = [row for row in rows if row["candidate"] == candidate]
            print(f"{candidate:<14} {len(items):>5} {np.mean([float(x['psnr']) for x in items]):>10.4f} {np.mean([float(x['ssim']) for x in items]):>10.4f} {np.mean([float(x['absdiff']) for x in items]):>10.4f}")
    print(f"output {args.out_dir}")


if __name__ == "__main__":
    main()

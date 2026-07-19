"""Train a compact history-to-future backward-flow predictor and evaluate BDD25."""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

from baseline.frame_io import import_cv2_np, read_rgb
from baseline.generators import compute_flow, warp_rgb
from baseline.metrics import mean_absdiff, psnr, ssim_gray
from baseline.refiner_dataset import filter_rows, load_split_rows, sample_rows
from baseline.refiner_model import import_torch
from baseline.video_io import extract_val_history_and_target


def resize_flow(cv2, flow, width: int, height: int):
    source_height, source_width = flow.shape[:2]
    output = cv2.resize(flow, (width, height), interpolation=cv2.INTER_LINEAR)
    output[:, :, 0] *= width / float(source_width)
    output[:, :, 1] *= height / float(source_height)
    return output


def make_features(cv2, np, previous, last, horizon: int, phase: str, width: int, height: int, max_horizon: int):
    previous = cv2.resize(previous, (width, height), interpolation=cv2.INTER_AREA)
    last = cv2.resize(last, (width, height), interpolation=cv2.INTER_AREA)
    history_flow = compute_flow(cv2, np, previous, last, width)
    flow_scale = max(8.0, float(max(width, height)) * 0.25)
    flow_channels = np.clip(history_flow / flow_scale, -1.0, 1.0)
    delta = np.mean(np.abs(last.astype(np.float32) - previous.astype(np.float32)), axis=2, keepdims=True) / 255.0
    context = np.zeros((height, width, 6), dtype=np.float32)
    context[:, :, 0] = min(1.0, horizon / float(max_horizon))
    try:
        phase_index = int(phase)
    except (TypeError, ValueError):
        phase_index = -1
    if 0 <= phase_index < 5:
        context[:, :, 1 + phase_index] = 1.0
    features = np.concatenate([last.astype(np.float32) / 255.0, flow_channels, delta, context], axis=2)
    return features, last, flow_scale


def build_model(nn, in_channels: int = 12, width: int = 48):
    class ResidualBlock(nn.Module):
        def __init__(self, channels):
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv2d(channels, channels, 3, padding=1),
                nn.SiLU(inplace=True),
                nn.Conv2d(channels, channels, 3, padding=1),
            )
            self.act = nn.SiLU(inplace=True)

        def forward(self, x):
            return self.act(x + self.net(x))

    class FlowPredictor(nn.Module):
        def __init__(self):
            super().__init__()
            self.stem = nn.Sequential(nn.Conv2d(in_channels, width, 5, padding=2), nn.SiLU(inplace=True))
            self.body = nn.Sequential(*[ResidualBlock(width) for _ in range(6)])
            self.head = nn.Sequential(nn.Conv2d(width, width, 3, padding=1), nn.SiLU(inplace=True), nn.Conv2d(width, 2, 3, padding=1), nn.Tanh())

        def forward(self, x):
            return self.head(self.body(self.stem(x)))

    return FlowPredictor()


class FlowDataset:
    def __init__(self, cv2, np, torch, rows, args, train: bool):
        self.cv2, self.np, self.torch = cv2, np, torch
        self.rows, self.args, self.train = rows, args, train
        self.rng = random.Random(args.seed + (100 if train else 200))

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        try:
            row = self.rows[index]
            horizon = self.rng.randint(1, self.args.max_horizon) if self.train else self.args.eval_horizon
            frames, target, meta = extract_val_history_and_target(self.cv2, row, 4, horizon)
            if not frames or len(frames) < 2 or target is None or (meta and "error" in meta):
                return None
            features, last, flow_scale = make_features(
                self.cv2, self.np, frames[-2], frames[-1], horizon, row.get("phase", ""),
                self.args.width, self.args.height, self.args.max_horizon,
            )
            target = self.cv2.resize(target, (self.args.width, self.args.height), interpolation=self.cv2.INTER_AREA)
            target_flow = compute_flow(self.cv2, self.np, target, last, self.args.width)
            target_flow = self.np.clip(target_flow / flow_scale, -1.0, 1.0)
            return (
                self.torch.from_numpy(features).permute(2, 0, 1).float(),
                self.torch.from_numpy(target_flow).permute(2, 0, 1).float(),
            )
        except (RuntimeError, OSError, ValueError):
            return None


def collate(batch):
    valid = [item for item in batch if item is not None]
    if not valid:
        return None
    import torch
    return torch.stack([item[0] for item in valid]), torch.stack([item[1] for item in valid])


def train_model(torch, nn, F, cv2, np, args, device):
    rows = filter_rows(load_split_rows(Path(args.index_dir), "train"), view_type="bdd")
    rows = sample_rows(rows, args.max_train_rows, args.seed)
    dataset = FlowDataset(cv2, np, torch, rows, args, train=True)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
        pin_memory=True, collate_fn=collate,
    )
    model = build_model(nn, width=args.model_width).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    for epoch in range(1, args.epochs + 1):
        model.train()
        total, count = 0.0, 0
        for batch in loader:
            if batch is None:
                continue
            inputs, targets = [item.to(device) for item in batch]
            predictions = model(inputs)
            flow_l1 = F.smooth_l1_loss(predictions, targets)
            grad = (predictions[:, :, :, 1:] - predictions[:, :, :, :-1] - targets[:, :, :, 1:] + targets[:, :, :, :-1]).abs().mean()
            grad += (predictions[:, :, 1:, :] - predictions[:, :, :-1, :] - targets[:, :, 1:, :] + targets[:, :, :-1, :]).abs().mean()
            loss = flow_l1 + 0.05 * grad
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total += float(loss.detach().cpu()) * inputs.shape[0]
            count += inputs.shape[0]
        print(f"epoch {epoch}/{args.epochs} | flow_loss={total / max(1, count):.6f} | samples={count}", flush=True)
    return model, len(rows)


def load_manifest(path: Path):
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_images(cv2, paths):
    return [image for path in paths if (image := read_rgb(cv2, Path(path))) is not None]


def sequence_score(cv2, np, predictions, targets):
    values = [(psnr(cv2, np, p, t), ssim_gray(cv2, np, p, t), mean_absdiff(np, p, t)) for p, t in zip(predictions, targets)]
    return tuple(float(np.mean([value[index] for value in values])) for index in range(3))


def evaluate(torch, cv2, np, model, args, device):
    rows_out = []
    model.eval()
    for position, row in enumerate(load_manifest(Path(args.manifest)), start=1):
        if row.get("view_type") != "bdd":
            continue
        paths = row.get("frames") or {}
        conditions = load_images(cv2, paths.get("condition") or [])
        targets = load_images(cv2, paths.get("target") or [])
        if len(conditions) < 2 or not targets:
            continue
        indices = row.get("frame_indices") or {}
        last_condition = int(indices["condition"][-1])
        offsets = [max(1, int(index) - last_condition) for index in indices["target"]]
        predictions = {scale: [] for scale in args.eval_flow_scales}
        for horizon, target in zip(offsets, targets):
            features, last_small, flow_scale = make_features(
                cv2, np, conditions[-2], conditions[-1], horizon, row.get("phase", ""),
                args.width, args.height, args.max_horizon,
            )
            tensor = torch.from_numpy(features).permute(2, 0, 1).unsqueeze(0).float().to(device)
            with torch.no_grad():
                flow = model(tensor)[0].permute(1, 2, 0).cpu().numpy() * flow_scale
            full_last = conditions[-1]
            full_flow = resize_flow(cv2, flow, full_last.shape[1], full_last.shape[0])
            for scale in args.eval_flow_scales:
                prediction = warp_rgb(cv2, np, full_last, full_flow, -scale)
                if prediction.shape[:2] != target.shape[:2]:
                    prediction = cv2.resize(prediction, (target.shape[1], target.shape[0]), interpolation=cv2.INTER_CUBIC)
                predictions[scale].append(prediction)
        repeat = [conditions[-1]] * len(targets)
        candidates = [("repeat_last", repeat)]
        candidates.extend(
            (f"predicted_flow_{int(round(scale * 100)):03d}", predictions[scale])
            for scale in args.eval_flow_scales
        )
        for candidate, frames in candidates:
            p, s, d = sequence_score(cv2, np, frames, targets)
            rows_out.append({"sample_id": row["sample_id"], "phase": row.get("phase", ""), "candidate": candidate, "psnr": f"{p:.6f}", "ssim": f"{s:.6f}", "absdiff": f"{d:.6f}"})
        print(f"[{position}] {row['sample_id']} | ok", flush=True)
    return rows_out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index_dir", default="/workspace/experiments/index")
    parser.add_argument("--manifest", default="/workspace/experiments/wan_ticft/manifests/val_ticft_balanced_125_frames.jsonl")
    parser.add_argument("--out_dir", default="/workspace/experiments/wan_ticft/bdd_future_flow_predictor_smoke")
    parser.add_argument("--max_train_rows", type=int, default=1000)
    parser.add_argument("--width", type=int, default=320)
    parser.add_argument("--height", type=int, default=180)
    parser.add_argument("--max_horizon", type=int, default=45)
    parser.add_argument("--eval_horizon", type=int, default=20)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--model_width", type=int, default=48)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=19)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--eval_flow_scales", type=float, nargs="+", default=[0.25, 0.5, 0.75, 1.0])
    args = parser.parse_args()

    cv2, np = import_cv2_np()
    torch, nn, F = import_torch()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"device={device} size={args.width}x{args.height} epochs={args.epochs}", flush=True)
    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint, map_location=device)
        model = build_model(nn, width=args.model_width).to(device)
        model.load_state_dict(checkpoint["model"])
        train_rows = int(checkpoint.get("train_rows", 0))
        print(f"loaded checkpoint {args.checkpoint}", flush=True)
    else:
        model, train_rows = train_model(torch, nn, F, cv2, np, args, device)
        torch.save({"model": model.state_dict(), "args": vars(args), "train_rows": train_rows}, out_dir / "future_flow_predictor.pt")
    rows = evaluate(torch, cv2, np, model, args, device)
    with (out_dir / "eval.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print()
    print(f"{'Candidate':<24} {'N':>5} {'PSNR':>10} {'SSIM':>10} {'AbsDiff':>10}")
    print("-" * 64)
    for candidate in ("repeat_last", *[f"predicted_flow_{int(round(scale * 100)):03d}" for scale in args.eval_flow_scales]):
        items = [row for row in rows if row["candidate"] == candidate]
        print(f"{candidate:<24} {len(items):>5} {np.mean([float(x['psnr']) for x in items]):>10.4f} {np.mean([float(x['ssim']) for x in items]):>10.4f} {np.mean([float(x['absdiff']) for x in items]):>10.4f}")
    print(f"train_rows={train_rows}")
    print(f"output {out_dir / 'eval.csv'}")


if __name__ == "__main__":
    main()

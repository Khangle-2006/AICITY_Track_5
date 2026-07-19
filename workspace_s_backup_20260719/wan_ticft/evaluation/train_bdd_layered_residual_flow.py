"""Train a residual-flow corrector on top of the BDD phase-router prediction."""

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
from wan_ticft.evaluation.evaluate_bdd_ego_motion import predict
from wan_ticft.evaluation.train_bdd_future_flow_predictor import build_model, resize_flow


def make_base(cv2, np, frames, horizon: int, history_gap: int, phase: str, estimate_width: int):
    predictions, _, _ = predict(
        cv2, np, frames, [horizon], history_gap, {"kind": "phase_router"}, estimate_width, phase
    )
    return predictions[0]


def make_features(cv2, np, previous, last, base, horizon: int, phase: str, width: int, height: int, max_horizon: int):
    previous = cv2.resize(previous, (width, height), interpolation=cv2.INTER_AREA)
    last = cv2.resize(last, (width, height), interpolation=cv2.INTER_AREA)
    base = cv2.resize(base, (width, height), interpolation=cv2.INTER_AREA)
    history_flow = compute_flow(cv2, np, previous, last, width)
    flow_scale = max(8.0, float(max(width, height)) * 0.25)
    flow_channels = np.clip(history_flow / flow_scale, -1.0, 1.0)
    delta = np.abs(base.astype(np.float32) - last.astype(np.float32)) / 255.0
    context = np.zeros((height, width, 6), dtype=np.float32)
    context[:, :, 0] = min(1.0, horizon / float(max_horizon))
    try:
        phase_index = int(phase)
    except (TypeError, ValueError):
        phase_index = -1
    if 0 <= phase_index < 5:
        context[:, :, 1 + phase_index] = 1.0
    features = np.concatenate(
        [
            previous.astype(np.float32) / 255.0,
            last.astype(np.float32) / 255.0,
            base.astype(np.float32) / 255.0,
            delta,
            flow_channels,
            context,
        ],
        axis=2,
    )
    return features, base, flow_scale


class LayeredDataset:
    def __init__(self, cv2, np, torch, rows, args):
        self.cv2, self.np, self.torch = cv2, np, torch
        self.rows, self.args = rows, args
        self.rng = random.Random(args.seed + 101)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        try:
            row = self.rows[index]
            horizon = self.rng.randint(1, self.args.max_horizon)
            frames, target, meta = extract_val_history_and_target(self.cv2, row, 4, horizon)
            if len(frames) < 2 or target is None or (meta and "error" in meta):
                return None
            history_gap = max(1, int(meta.get("history_gap_frames", 1)))
            base = make_base(
                self.cv2, self.np, frames, horizon, history_gap, row.get("phase", ""), self.args.width
            )
            features, base_small, flow_scale = make_features(
                self.cv2, self.np, frames[-2], frames[-1], base, horizon, row.get("phase", ""),
                self.args.width, self.args.height, self.args.max_horizon,
            )
            target = self.cv2.resize(target, (self.args.width, self.args.height), interpolation=self.cv2.INTER_AREA)
            residual = compute_flow(self.cv2, self.np, target, base_small, self.args.width)
            residual = self.np.clip(residual / flow_scale, -1.0, 1.0)
            return (
                self.torch.from_numpy(features).permute(2, 0, 1).float(),
                self.torch.from_numpy(residual).permute(2, 0, 1).float(),
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
    loader = torch.utils.data.DataLoader(
        LayeredDataset(cv2, np, torch, rows, args),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate,
    )
    model = build_model(nn, in_channels=20, width=args.model_width).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    for epoch in range(1, args.epochs + 1):
        model.train()
        total, count = 0.0, 0
        for batch in loader:
            if batch is None:
                continue
            inputs, targets = [item.to(device) for item in batch]
            outputs = model(inputs)
            flow_loss = F.smooth_l1_loss(outputs, targets)
            grad = (outputs[:, :, :, 1:] - outputs[:, :, :, :-1] - targets[:, :, :, 1:] + targets[:, :, :, :-1]).abs().mean()
            grad += (outputs[:, :, 1:, :] - outputs[:, :, :-1, :] - targets[:, :, 1:, :] + targets[:, :, :-1, :]).abs().mean()
            loss = flow_loss + 0.05 * grad
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total += float(loss.detach().cpu()) * inputs.shape[0]
            count += inputs.shape[0]
        print(f"epoch {epoch}/{args.epochs} | residual_flow_loss={total / max(count, 1):.6f} | samples={count}", flush=True)
    return model, len(rows)


def load_manifest(path: Path):
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_images(cv2, paths):
    return [image for path in paths if (image := read_rgb(cv2, Path(path))) is not None]


def score(cv2, np, frames, targets):
    metrics = [(psnr(cv2, np, p, t), ssim_gray(cv2, np, p, t), mean_absdiff(np, p, t)) for p, t in zip(frames, targets)]
    return tuple(float(np.mean([value[index] for value in metrics])) for index in range(3))


def evaluate(torch, cv2, np, model, args, device):
    output = []
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
        history_gap = max(1, int(indices["condition"][-1]) - int(indices["condition"][-2]))
        offsets = [max(1, int(index) - int(indices["condition"][-1])) for index in indices["target"]]
        bases, layered = [], {scale: [] for scale in args.eval_residual_scales}
        for horizon, target in zip(offsets, targets):
            base = make_base(cv2, np, conditions, horizon, history_gap, row.get("phase", ""), args.estimate_width)
            features, _, flow_scale = make_features(
                cv2, np, conditions[-2], conditions[-1], base, horizon, row.get("phase", ""),
                args.width, args.height, args.max_horizon,
            )
            tensor = torch.from_numpy(features).permute(2, 0, 1).unsqueeze(0).float().to(device)
            with torch.no_grad():
                residual = model(tensor)[0].permute(1, 2, 0).cpu().numpy() * flow_scale
            residual = resize_flow(cv2, residual, base.shape[1], base.shape[0])
            bases.append(base)
            for scale in args.eval_residual_scales:
                layered[scale].append(warp_rgb(cv2, np, base, residual, -scale))
        candidates = [("phase_router_v1", bases)]
        candidates.extend(
            (f"layered_residual_{int(round(scale * 100)):03d}", layered[scale])
            for scale in args.eval_residual_scales
        )
        for candidate, frames in candidates:
            p, s, d = score(cv2, np, frames, targets)
            output.append({"sample_id": row["sample_id"], "phase": row.get("phase", ""), "candidate": candidate, "psnr": f"{p:.6f}", "ssim": f"{s:.6f}", "absdiff": f"{d:.6f}"})
        print(f"[{position}] {row['sample_id']} | ok", flush=True)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index_dir", default="/workspace/experiments/index")
    parser.add_argument("--manifest", default="/workspace/experiments/wan_ticft/manifests/val_ticft_balanced_125_frames.jsonl")
    parser.add_argument("--out_dir", default="/workspace/experiments/wan_ticft/bdd_layered_residual_flow_smoke")
    parser.add_argument("--max_train_rows", type=int, default=1000)
    parser.add_argument("--width", type=int, default=320)
    parser.add_argument("--height", type=int, default=180)
    parser.add_argument("--estimate_width", type=int, default=640)
    parser.add_argument("--max_horizon", type=int, default=45)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--model_width", type=int, default=48)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--eval_residual_scales", type=float, nargs="+", default=[0.25, 0.5, 0.75, 1.0])
    args = parser.parse_args()

    cv2, np = import_cv2_np()
    torch, nn, F = import_torch()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"device={device} size={args.width}x{args.height} epochs={args.epochs}", flush=True)
    model, train_rows = train_model(torch, nn, F, cv2, np, args, device)
    torch.save({"model": model.state_dict(), "args": vars(args), "train_rows": train_rows}, out_dir / "layered_residual_flow.pt")
    rows = evaluate(torch, cv2, np, model, args, device)
    with (out_dir / "eval.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print()
    print(f"{'Candidate':<26} {'N':>5} {'PSNR':>10} {'SSIM':>10} {'AbsDiff':>10}")
    print("-" * 66)
    for candidate in ("phase_router_v1", *[f"layered_residual_{int(round(scale * 100)):03d}" for scale in args.eval_residual_scales]):
        items = [row for row in rows if row["candidate"] == candidate]
        print(f"{candidate:<26} {len(items):>5} {np.mean([float(x['psnr']) for x in items]):>10.4f} {np.mean([float(x['ssim']) for x in items]):>10.4f} {np.mean([float(x['absdiff']) for x in items]):>10.4f}")
    print(f"train_rows={train_rows}")
    print(f"output {out_dir / 'eval.csv'}")


if __name__ == "__main__":
    main()

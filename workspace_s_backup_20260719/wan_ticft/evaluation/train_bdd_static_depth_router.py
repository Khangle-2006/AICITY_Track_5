from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from baseline.frame_io import import_cv2_np, read_rgb
from baseline.generators import compute_flow
from wan_ticft.evaluation.evaluate_video_reports import evaluate_candidate, read_report


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_images(cv2_module, paths: list[str]):
    return [image for raw in paths if (image := read_rgb(cv2_module, Path(raw))) is not None]


def load_video_indices(cv2_module, row: dict[str, Any], indices: list[int]):
    video_path = Path(str(row.get("video_path") or ""))
    if not video_path.exists() or not indices:
        return []
    capture = cv2_module.VideoCapture(str(video_path))
    if not capture.isOpened():
        return []
    frames = []
    for index in indices:
        capture.set(cv2_module.CAP_PROP_POS_FRAMES, int(index))
        ok, frame = capture.read()
        if not ok or frame is None:
            capture.release()
            return []
        frames.append(cv2_module.cvtColor(frame, cv2_module.COLOR_BGR2RGB))
    capture.release()
    return frames


def load_conditions(cv2_module, row: dict[str, Any]):
    paths = row.get("frames") or {}
    conditions = load_images(cv2_module, paths.get("condition") or [])
    if conditions:
        return conditions
    indices = row.get("frame_indices") or {}
    return load_video_indices(cv2_module, row, indices.get("condition") or [])


def prompt_features(prompt: str) -> list[float]:
    text = prompt.lower()
    speeds = [float(value) for value in re.findall(r"(\d+(?:\.\d+)?)\s*km/?h", text)]
    speed = speeds[0] if speeds else 0.0
    return [
        min(speed, 40.0) / 40.0,
        1.0 if "stationary" in text or "0km" in text or "0 km" in text else 0.0,
        1.0 if "standing" in text else 0.0,
        1.0 if "crossing" in text else 0.0,
        1.0 if "turning" in text else 0.0,
        1.0 if "moving" in text else 0.0,
    ]


def row_features(cv2_module, np_module, row: dict[str, Any], flow_width: int) -> np.ndarray | None:
    conditions = load_conditions(cv2_module, row)
    if len(conditions) < 2:
        return None
    resized = [
        cv2_module.resize(frame, (160, 90), interpolation=cv2_module.INTER_AREA).astype(np.float32)
        for frame in conditions
    ]
    diffs = [float(np_module.mean(np_module.abs(b - a)) / 255.0) for a, b in zip(resized[:-1], resized[1:])]
    last = conditions[-1]
    prev = conditions[-2]
    flow = compute_flow(cv2_module, np_module, prev, last, flow_width)
    mag = np_module.sqrt(flow[:, :, 0] ** 2 + flow[:, :, 1] ** 2)
    gray = cv2_module.cvtColor(cv2_module.resize(last, (160, 90), interpolation=cv2_module.INTER_AREA), cv2_module.COLOR_RGB2GRAY)
    lap_var = float(cv2_module.Laplacian(gray, cv2_module.CV_32F).var()) / 1000.0
    indices = row.get("frame_indices") or {}
    condition = indices.get("condition") or []
    target = indices.get("target") or []
    history_gap = float(int(condition[-1]) - int(condition[-2])) if len(condition) >= 2 else 0.0
    horizon = float(int(target[-1]) - int(condition[-1])) if condition and target else 0.0
    phase_raw = int(row.get("phase") or 0)
    phase = [1.0 if phase_raw == index else 0.0 for index in range(5)]
    values = [
        *phase,
        len(conditions) / 8.0,
        history_gap / 40.0,
        horizon / 120.0,
        float(np_module.mean(diffs)) if diffs else 0.0,
        float(np_module.std(diffs)) if diffs else 0.0,
        float(diffs[-1]) if diffs else 0.0,
        float(np_module.mean(mag)) / 10.0,
        float(np_module.percentile(mag, 75)) / 10.0,
        float(np_module.percentile(mag, 95)) / 10.0,
        lap_var,
        *prompt_features(str(row.get("prompt") or "")),
    ]
    return np_module.asarray(values, dtype=np.float32)


def score_rows(cv2_module, np_module, rows, depth_report, repeat_report, offset_mode: str):
    scores: dict[str, dict[str, float]] = {}
    for row in rows:
        sample_id = row["sample_id"]
        depth = evaluate_candidate(cv2_module, np_module, row, "depth", depth_report.get(sample_id), offset_mode)
        repeat = evaluate_candidate(cv2_module, np_module, row, "repeat", repeat_report.get(sample_id), offset_mode)
        if depth["status"] == "ok" and repeat["status"] == "ok":
            scores[sample_id] = {
                "depth_absdiff": float(depth["absdiff"]),
                "repeat_absdiff": float(repeat["absdiff"]),
                "depth_psnr": float(depth["psnr"]),
                "repeat_psnr": float(repeat["psnr"]),
            }
    return scores


class Router(nn.Module):
    def __init__(self, in_features: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, 32),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(32, 16),
            nn.SiLU(),
            nn.Linear(16, 2),
        )

    def forward(self, value):
        return self.net(value)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train BDD static router: depth_a75 vs repeat_last.")
    parser.add_argument("--train_manifest", type=Path, required=True)
    parser.add_argument("--train_depth_report", type=Path, required=True)
    parser.add_argument("--train_repeat_report", type=Path, required=True)
    parser.add_argument("--eval_manifest", type=Path, required=True)
    parser.add_argument("--eval_depth_report", type=Path, required=True)
    parser.add_argument("--eval_repeat_report", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--train_limit", type=int, default=48)
    parser.add_argument("--flow_width", type=int, default=426)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--repeat_margin", type=float, default=0.25, help="Require repeat to beat depth by this AbsDiff margin.")
    parser.add_argument("--repeat_threshold", type=float, default=0.50)
    parser.add_argument("--seed", type=int, default=20260625)
    parser.add_argument("--clean", action="store_true")
    args = parser.parse_args()

    cv2_module, np_module = import_cv2_np()
    if args.clean and args.out_dir.exists():
        shutil.rmtree(args.out_dir)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "args.json").write_text(json.dumps(vars(args), indent=2, default=str), encoding="utf-8")

    train_rows = [row for row in read_jsonl(args.train_manifest) if row.get("view_type") == "bdd"][: args.train_limit]
    eval_rows = [row for row in read_jsonl(args.eval_manifest) if row.get("view_type") == "bdd"]
    train_depth = read_report(args.train_depth_report)
    train_repeat = read_report(args.train_repeat_report)
    eval_depth = read_report(args.eval_depth_report)
    eval_repeat = read_report(args.eval_repeat_report)
    train_scores = score_rows(cv2_module, np_module, train_rows, train_depth, train_repeat, "timestamp")

    features = []
    labels = []
    train_feature_rows = []
    for row in train_rows:
        sample_id = row["sample_id"]
        score = train_scores.get(sample_id)
        feat = row_features(cv2_module, np_module, row, args.flow_width)
        if score is None or feat is None:
            continue
        repeat_better = score["repeat_absdiff"] + args.repeat_margin < score["depth_absdiff"]
        label = 1 if repeat_better else 0
        features.append(feat)
        labels.append(label)
        train_feature_rows.append(
            {
                "sample_id": sample_id,
                "phase": row.get("phase", ""),
                "label": "repeat" if label else "depth",
                **score,
            }
        )
    if len(features) < 8:
        raise RuntimeError(f"not enough train features: {len(features)}")

    torch.manual_seed(args.seed)
    x = torch.from_numpy(np_module.stack(features)).float()
    y = torch.tensor(labels).long()
    mean = x.mean(dim=0, keepdim=True)
    std = x.std(dim=0, keepdim=True).clamp_min(1e-4)
    x_norm = (x - mean) / std
    weights = torch.tensor(
        [1.0 / max(1, int((y == 0).sum())), 1.0 / max(1, int((y == 1).sum()))],
        dtype=torch.float32,
    )
    weights = weights / weights.mean()
    model = Router(x.shape[1])
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
    for epoch in range(args.epochs):
        logits = model(x_norm)
        loss = F.cross_entropy(logits, y, weight=weights)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    eval_choices = []
    report_rows = []
    for row in eval_rows:
        sample_id = row["sample_id"]
        feat = row_features(cv2_module, np_module, row, args.flow_width)
        if feat is None:
            chosen = "depth"
            repeat_prob = 0.0
        else:
            value = (torch.from_numpy(feat).float()[None] - mean) / std
            with torch.no_grad():
                probs = model(value).softmax(dim=1)[0]
            repeat_prob = float(probs[1].item())
            chosen = "repeat" if repeat_prob >= args.repeat_threshold else "depth"
        report = eval_repeat if chosen == "repeat" else eval_depth
        video = report.get(sample_id)
        if video is None:
            chosen = "depth"
            video = eval_depth.get(sample_id)
        if video is None:
            continue
        report_rows.append({"sample_id": sample_id, "video": str(video), "status": "ok", "chosen": chosen, "repeat_prob": f"{repeat_prob:.6f}"})
        eval_choices.append({"sample_id": sample_id, "phase": row.get("phase", ""), "chosen": chosen, "repeat_prob": f"{repeat_prob:.6f}"})
        print(f"{sample_id} phase={row.get('phase')} chosen={chosen} repeat_prob={repeat_prob:.3f}", flush=True)

    write_csv(args.out_dir / "train_labels.csv", train_feature_rows)
    write_csv(args.out_dir / "choices.csv", eval_choices)
    write_csv(args.out_dir / "routed_report.csv", report_rows)
    torch.save({"model": model.state_dict(), "mean": mean, "std": std, "args": vars(args)}, args.out_dir / "router.pt")
    print(f"train_rows={len(features)} repeat_labels={sum(labels)} depth_labels={len(labels) - sum(labels)}")
    print(f"output={args.out_dir / 'routed_report.csv'}")


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path
from typing import Any

from baseline.frame_io import import_cv2_np, read_rgb
from wan_ticft.evaluation.evaluate_video_reports import (
    evaluate_candidate,
    map_target_offsets,
    read_report,
    read_video_rgb_frames,
    video_fps,
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def parse_report(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise ValueError(f"report must be NAME=CSV, got {value!r}")
    name, path = value.split("=", 1)
    return name, Path(path)


def load_candidate_frames(cv2, row: dict[str, Any], report: dict[str, Path]):
    video = report.get(row["sample_id"])
    if video is None:
        return [], None
    mapped = map_target_offsets(row, video_fps(cv2, video))
    frame_map = read_video_rgb_frames(cv2, video, mapped)
    return [frame_map[index] for index in mapped if index in frame_map], video


def load_condition(cv2, row: dict[str, Any]):
    paths = (row.get("frames") or {}).get("condition") or []
    return [img for path in paths if (img := read_rgb(cv2, Path(path))) is not None]


def feature_for(cv2, np, row: dict[str, Any], frames, video):
    cond = load_condition(cv2, row)
    if not frames or not cond:
        return None
    first = cv2.resize(frames[0], (160, 90), interpolation=cv2.INTER_AREA).astype(np.float32)
    last = cv2.resize(frames[-1], (160, 90), interpolation=cv2.INTER_AREA).astype(np.float32)
    anchor = cv2.resize(cond[-1], (160, 90), interpolation=cv2.INTER_AREA).astype(np.float32)
    diffs = []
    prev = None
    for frame in frames:
        small = cv2.resize(frame, (160, 90), interpolation=cv2.INTER_AREA).astype(np.float32)
        if prev is not None:
            diffs.append(np.abs(small - prev).mean() / 255.0)
        prev = small
    gray = [cv2.cvtColor(cv2.resize(frame, (160, 90), interpolation=cv2.INTER_AREA), cv2.COLOR_RGB2GRAY) for frame in frames[::4]]
    sharp = [float(cv2.Laplacian(g, cv2.CV_32F).var()) / 1000.0 for g in gray]
    phase = float(row.get("phase") or 0) / 4.0
    return np.array(
        [
            phase,
            len(frames) / 40.0,
            float(video_fps(cv2, video) or 30.0) / 30.0,
            np.abs(first - anchor).mean() / 255.0,
            np.abs(last - anchor).mean() / 255.0,
            np.mean(diffs) if diffs else 0.0,
            np.std(diffs) if diffs else 0.0,
            np.mean(sharp) if sharp else 0.0,
            np.std(sharp) if sharp else 0.0,
        ],
        dtype=np.float32,
    )


def compute_scores(cv2, np, rows, reports, offset_mode: str):
    scores = {}
    for row in rows:
        sample_id = row["sample_id"]
        scores[sample_id] = {}
        for name, report in reports.items():
            result = evaluate_candidate(cv2, np, row, name, report.get(sample_id), offset_mode)
            if result["status"] == "ok":
                scores[sample_id][name] = {
                    "psnr": float(result["psnr"]),
                    "ssim": float(result["ssim"]),
                    "absdiff": float(result["absdiff"]),
                }
    return scores


def choose_labels(scores, alpha: float, beta: float):
    labels = {}
    for sample_id, values in scores.items():
        if not values:
            continue
        labels[sample_id] = max(
            values,
            key=lambda name: values[name]["psnr"] + alpha * values[name]["ssim"] - beta * values[name]["absdiff"],
        )
    return labels


def main() -> None:
    parser = argparse.ArgumentParser(description="Train held-out supervised candidate router.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--report", action="append", required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--train_rows", type=int, default=16)
    parser.add_argument("--alpha_ssim", type=float, default=5.0)
    parser.add_argument("--beta_absdiff", type=float, default=0.02)
    parser.add_argument("--default", default="mix65")
    parser.add_argument("--clean", action="store_true")
    args = parser.parse_args()

    cv2, np = import_cv2_np()
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    rows = [row for row in read_jsonl(args.manifest) if row.get("view_type") == "bdd"]
    reports = {name: read_report(path) for name, path in map(parse_report, args.report)}
    labels_order = list(reports.keys())
    if args.default not in reports:
        raise ValueError(f"default {args.default!r} not in reports")
    if args.clean and args.out_dir.exists():
        shutil.rmtree(args.out_dir)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    scores = compute_scores(cv2, np, rows, reports, "timestamp")
    oracle_labels = choose_labels(scores, args.alpha_ssim, args.beta_absdiff)
    train_set = rows[: args.train_rows]
    eval_set = rows[args.train_rows :]

    xs, ys = [], []
    for row in train_set:
        sample_id = row["sample_id"]
        label = oracle_labels.get(sample_id)
        if label is None:
            continue
        feats = []
        ok = True
        for name in labels_order:
            frames, video = load_candidate_frames(cv2, row, reports[name])
            feat = feature_for(cv2, np, row, frames, video) if video is not None else None
            if feat is None:
                ok = False
                break
            feats.append(feat)
        if ok:
            xs.append(np.concatenate(feats))
            ys.append(labels_order.index(label))
    if not xs:
        raise RuntimeError("no train features")
    x = torch.from_numpy(np.stack(xs)).float()
    y = torch.tensor(ys).long()
    mean = x.mean(dim=0, keepdim=True)
    std = x.std(dim=0, keepdim=True).clamp_min(1e-4)
    x = (x - mean) / std

    model = nn.Sequential(nn.Linear(x.shape[1], 64), nn.ReLU(), nn.Dropout(0.1), nn.Linear(64, len(labels_order)))
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-3)
    for epoch in range(200):
        logits = model(x)
        loss = F.cross_entropy(logits, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

    choice_rows = []
    routed_rows = []
    for row in rows:
        sample_id = row["sample_id"]
        feats = []
        ok = True
        videos = {}
        for name in labels_order:
            frames, video = load_candidate_frames(cv2, row, reports[name])
            videos[name] = video
            feat = feature_for(cv2, np, row, frames, video) if video is not None else None
            if feat is None:
                ok = False
                break
            feats.append(feat)
        if ok:
            feat_tensor = (torch.from_numpy(np.concatenate(feats)).float()[None] - mean) / std
            pred = int(model(feat_tensor).argmax(dim=1).item())
            chosen = labels_order[pred]
        else:
            chosen = args.default
        video = videos.get(chosen) or reports[args.default].get(sample_id)
        routed_rows.append({"sample_id": sample_id, "video": str(video), "status": "ok", "chosen": chosen})
        choice_rows.append(
            {
                "sample_id": sample_id,
                "split": "train" if row in train_set else "eval",
                "chosen": chosen,
                "oracle": oracle_labels.get(sample_id, ""),
                "chosen_psnr": scores.get(sample_id, {}).get(chosen, {}).get("psnr", ""),
                "oracle_psnr": scores.get(sample_id, {}).get(oracle_labels.get(sample_id, ""), {}).get("psnr", ""),
            }
        )
        print(f"{sample_id} split={choice_rows[-1]['split']} chosen={chosen} oracle={choice_rows[-1]['oracle']}", flush=True)

    with (args.out_dir / "routed_report.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["sample_id", "video", "status", "chosen"])
        writer.writeheader()
        writer.writerows(routed_rows)
    with (args.out_dir / "choices.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["sample_id", "split", "chosen", "oracle", "chosen_psnr", "oracle_psnr"])
        writer.writeheader()
        writer.writerows(choice_rows)
    torch.save({"model": model.state_dict(), "mean": mean, "std": std, "labels": labels_order}, args.out_dir / "router.pt")
    print(f"output={args.out_dir / 'routed_report.csv'}", flush=True)


if __name__ == "__main__":
    main()

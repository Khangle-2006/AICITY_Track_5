from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from baseline.frame_io import import_cv2_np, read_rgb
from wan_ticft.evaluation.evaluate_video_reports import map_target_offsets, read_report, read_video_rgb_frames, video_fps


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_targets(cv2, row: dict[str, Any]):
    return [img for path in (row.get("frames") or {}).get("target", []) if (img := read_rgb(cv2, Path(path))) is not None]


def load_candidate(cv2, row: dict[str, Any], report: dict[str, Path]):
    video = report.get(row["sample_id"])
    if video is None:
        return [], None
    mapped = map_target_offsets(row, video_fps(cv2, video))
    frame_map = read_video_rgb_frames(cv2, video, mapped)
    return [frame_map[index] for index in mapped if index in frame_map], video


def parse_report(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise ValueError(f"report must be name=csv, got {value!r}")
    name, path = value.split("=", 1)
    return name, Path(path)


def covariance_score(torch, values, mean, std):
    z = (values - mean) / std.clamp_min(1e-4)
    return z.pow(2).mean()


def main() -> None:
    parser = argparse.ArgumentParser(description="No-reference per-sample rerank using real-video feature distribution.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--report", action="append", required=True, help="NAME=CSV")
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--view_type", default="bdd")
    parser.add_argument("--clip_frames", type=int, default=16)
    parser.add_argument("--frame_stride", type=int, default=4)
    parser.add_argument("--image_weight", type=float, default=1.0)
    parser.add_argument("--video_weight", type=float, default=1.0)
    parser.add_argument("--smoothness_weight", type=float, default=0.05)
    args = parser.parse_args()

    cv2, np = import_cv2_np()
    import torch
    import torch.nn.functional as F
    from torchvision.models import Inception_V3_Weights, inception_v3
    from torchvision.models.video import R3D_18_Weights, r3d_18

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image_model = inception_v3(weights=Inception_V3_Weights.DEFAULT)
    image_model.fc = torch.nn.Identity()
    image_model.eval().to(device)
    video_model = r3d_18(weights=R3D_18_Weights.DEFAULT)
    video_model.fc = torch.nn.Identity()
    video_model.eval().to(device)
    image_mean = torch.tensor((0.485, 0.456, 0.406), device=device)[None, :, None, None]
    image_std = torch.tensor((0.229, 0.224, 0.225), device=device)[None, :, None, None]
    video_mean = torch.tensor((0.43216, 0.394666, 0.37645), device=device)[None, :, None, None, None]
    video_std = torch.tensor((0.22803, 0.22145, 0.216989), device=device)[None, :, None, None, None]

    def image_feature(frame):
        tensor = torch.from_numpy(frame.copy()).permute(2, 0, 1).float().div(255).unsqueeze(0).to(device)
        tensor = F.interpolate(tensor, size=(299, 299), mode="bilinear", align_corners=False)
        with torch.no_grad():
            return image_model((tensor - image_mean) / image_std).cpu()

    def video_feature(frames):
        indices = np.linspace(0, len(frames) - 1, args.clip_frames).round().astype(int)
        tensor = torch.stack(
            [torch.from_numpy(frames[index].copy()).permute(2, 0, 1).float().div(255) for index in indices]
        )
        tensor = F.interpolate(tensor, size=(112, 112), mode="bilinear", align_corners=False)
        tensor = tensor.permute(1, 0, 2, 3).unsqueeze(0).to(device)
        with torch.no_grad():
            return video_model((tensor - video_mean) / video_std).cpu()

    def temporal_smoothness(frames):
        if len(frames) < 2:
            return 0.0
        diffs = []
        prev = None
        for frame in frames:
            small = cv2.resize(frame, (160, 90), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
            if prev is not None:
                diffs.append(float(np.mean(np.abs(small - prev))))
            prev = small
        return float(np.mean(diffs)) if diffs else 0.0

    rows = [row for row in read_jsonl(args.manifest) if row.get("view_type") == args.view_type]
    reports = {name: read_report(path) for name, path in map(parse_report, args.report)}

    target_image_features = []
    target_video_features = []
    valid_rows = []
    for pos, row in enumerate(rows, start=1):
        targets = load_targets(cv2, row)
        if len(targets) < args.clip_frames:
            continue
        for frame in targets[:: args.frame_stride]:
            target_image_features.append(image_feature(frame))
        target_video_features.append(video_feature(targets))
        valid_rows.append(row)
        print(f"reference [{pos}/{len(rows)}] {row['sample_id']} | ok", flush=True)
    if not target_image_features or not target_video_features:
        raise RuntimeError("no reference features")
    ref_img = torch.cat(target_image_features)
    ref_vid = torch.cat(target_video_features)
    img_mean, img_std = ref_img.mean(dim=0), ref_img.std(dim=0, unbiased=False).clamp_min(1e-4)
    vid_mean, vid_std = ref_vid.mean(dim=0), ref_vid.std(dim=0, unbiased=False).clamp_min(1e-4)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    routed_rows = []
    choice_rows = []
    for row in valid_rows:
        choices = []
        for name, report in reports.items():
            frames, video = load_candidate(cv2, row, report)
            if video is None or len(frames) != len(load_targets(cv2, row)):
                continue
            img_feats = torch.cat([image_feature(frame) for frame in frames[:: args.frame_stride]])
            vid_feat = video_feature(frames)
            img_score = covariance_score(torch, img_feats, img_mean, img_std)
            vid_score = covariance_score(torch, vid_feat, vid_mean, vid_std)
            smooth = temporal_smoothness(frames)
            score = args.image_weight * float(img_score) + args.video_weight * float(vid_score) + args.smoothness_weight * smooth
            choices.append((score, name, video, float(img_score), float(vid_score), smooth))
        if not choices:
            continue
        choices.sort(key=lambda item: item[0])
        score, name, video, img_score, vid_score, smooth = choices[0]
        routed_rows.append({"sample_id": row["sample_id"], "video": str(video), "status": "ok", "chosen": name})
        choice_rows.append(
            {
                "sample_id": row["sample_id"],
                "chosen": name,
                "score": f"{score:.6f}",
                "image_score": f"{img_score:.6f}",
                "video_score": f"{vid_score:.6f}",
                "smoothness": f"{smooth:.6f}",
                "ranked": ";".join(f"{candidate}:{candidate_score:.4f}" for candidate_score, candidate, *_rest in choices),
            }
        )
        print(f"choose {row['sample_id']} -> {name} score={score:.4f}", flush=True)

    with (args.out_dir / "routed_report.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["sample_id", "video", "status", "chosen"])
        writer.writeheader()
        writer.writerows(routed_rows)
    with (args.out_dir / "choices.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["sample_id", "chosen", "score", "image_score", "video_score", "smoothness", "ranked"])
        writer.writeheader()
        writer.writerows(choice_rows)
    print(f"output={args.out_dir / 'routed_report.csv'}", flush=True)


if __name__ == "__main__":
    main()

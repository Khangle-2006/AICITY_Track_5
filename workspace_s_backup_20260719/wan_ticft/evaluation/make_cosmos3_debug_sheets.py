from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from baseline.frame_io import import_cv2_np, read_rgb


def read_manifest(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def read_report(path: Path) -> dict[str, Path]:
    out = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("status") == "ok":
                out[row["sample_id"]] = Path(row["video"])
    return out


def read_video_frame(cv2, path: Path, index: int):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return None
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, index))
    ok, bgr = cap.read()
    cap.release()
    if not ok or bgr is None:
        return None
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def resize(cv2, img, width: int, height: int):
    if img is None:
        return None
    if img.shape[1] == width and img.shape[0] == height:
        return img
    return cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)


def put_label(cv2, img, text: str):
    out = img.copy()
    cv2.rectangle(out, (0, 0), (out.shape[1], 34), (0, 0, 0), -1)
    cv2.putText(out, text, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--cosmos_report", type=Path, required=True)
    parser.add_argument("--transition_report", type=Path, default=Path("/workspace/experiments/wan_ticft/ltxv_bdd25_gate/report.csv"))
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--offsets", type=int, nargs="+", default=[1, 8, 16, 24, 32, 40])
    parser.add_argument("--thumb_w", type=int, default=320)
    parser.add_argument("--thumb_h", type=int, default=180)
    args = parser.parse_args()

    cv2, np = import_cv2_np()
    rows = read_manifest(args.manifest)
    cosmos = read_report(args.cosmos_report)
    transition = read_report(args.transition_report)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    for row in rows:
        sample_id = row["sample_id"]
        condition_paths = (row.get("frames") or {}).get("condition") or []
        target_paths = (row.get("frames") or {}).get("target") or []
        panels = []
        if condition_paths:
            img = resize(cv2, read_rgb(cv2, Path(condition_paths[-1])), args.thumb_w, args.thumb_h)
            panels.append(put_label(cv2, img, "last condition"))
        for offset in args.offsets:
            target_idx = min(max(0, offset - 1), len(target_paths) - 1)
            target = resize(cv2, read_rgb(cv2, Path(target_paths[target_idx])), args.thumb_w, args.thumb_h)
            trans = resize(cv2, read_video_frame(cv2, transition[sample_id], offset), args.thumb_w, args.thumb_h)
            cos = resize(cv2, read_video_frame(cv2, cosmos[sample_id], max(0, round(offset * 24 / 30))), args.thumb_w, args.thumb_h)
            panels.extend(
                [
                    put_label(cv2, target, f"target +{offset}"),
                    put_label(cv2, trans, f"ltx +{offset}"),
                    put_label(cv2, cos, f"cosmos +{offset}"),
                ]
            )
        rows_img = []
        for start in range(0, len(panels), 3):
            chunk = panels[start : start + 3]
            if len(chunk) < 3:
                pad = np.zeros_like(chunk[0])
                while len(chunk) < 3:
                    chunk.append(pad)
            rows_img.append(np.concatenate(chunk, axis=1))
        sheet = np.concatenate(rows_img, axis=0)
        out = args.out_dir / f"{sample_id}.jpg"
        cv2.imwrite(str(out), cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR))
        print(out)


if __name__ == "__main__":
    main()

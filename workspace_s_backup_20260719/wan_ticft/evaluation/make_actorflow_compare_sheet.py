"""Make a compact MF/actorflow/GT comparison sheet for selected BDD samples."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from baseline.frame_io import import_cv2_np


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def read_frame(cv2, video: Path, index: int):
    cap = cv2.VideoCapture(str(video))
    frame = None
    for _ in range(index + 1):
        ok, bgr = cap.read()
        if not ok:
            break
        frame = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    cap.release()
    return frame


def target_offset(row: dict, target_position: int) -> int:
    indices = row.get("frame_indices") or {}
    condition = indices.get("condition") or []
    target = indices.get("target") or []
    if condition and target:
        return max(1, int(target[target_position]) - int(condition[-1]))
    return target_position


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--base_videos", type=Path, required=True)
    parser.add_argument("--candidate_videos", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--sample_id", action="append", required=True)
    parser.add_argument("--width", type=int, default=320)
    parser.add_argument("--height", type=int, default=180)
    args = parser.parse_args()

    cv2, np = import_cv2_np()
    rows = {row["sample_id"]: row for row in read_jsonl(args.manifest)}
    strips = []
    for sid in args.sample_id:
        row = rows[sid]
        targets = (row.get("frames") or {}).get("target") or []
        if not targets:
            continue
        target_position = len(targets) // 2
        offset = target_offset(row, target_position)
        target_bgr = cv2.imread(str(targets[target_position]), cv2.IMREAD_COLOR)
        base = read_frame(cv2, args.base_videos / f"{sid}.avi", offset)
        candidate = read_frame(cv2, args.candidate_videos / f"{sid}.avi", offset)
        if target_bgr is None or base is None or candidate is None:
            continue
        target = cv2.cvtColor(target_bgr, cv2.COLOR_BGR2RGB)
        cells = []
        for label, image in (("MF", base), ("SEM", candidate), ("GT", target)):
            cell = cv2.resize(image, (args.width, args.height), interpolation=cv2.INTER_AREA)
            cv2.putText(cell, label, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
            cells.append(cell)
        strip = np.concatenate(cells, axis=1)
        cv2.putText(strip, sid[-34:], (8, args.height - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
        strips.append(strip)
    if not strips:
        raise RuntimeError("no strips written")
    sheet = np.concatenate(strips, axis=0)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(args.out), cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR))
    print(args.out)


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from baseline.frame_io import import_cv2_np


ROIS = {
    "near_black_car_right": (515, 245, 790, 455),
    "center_red_car": (430, 210, 610, 360),
    "far_lane_cars": (250, 155, 520, 260),
    "road_markings": (260, 345, 640, 510),
}


def read_frame(cv2, root: Path, index: int):
    path = root / "video3806" / f"{index}.png"
    frame = cv2.imread(str(path))
    if frame is None:
        raise FileNotFoundError(path)
    return frame


def resize_to_audit(cv2, frame, size=(896, 512)):
    width, height = size
    if frame.shape[1] == width and frame.shape[0] == height:
        return frame
    return cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)


def flow_mag(cv2, np, first, second, roi):
    x1, y1, x2, y2 = roi
    a = cv2.cvtColor(first[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)
    b = cv2.cvtColor(second[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)
    flow = cv2.calcOpticalFlowFarneback(a, b, None, 0.5, 3, 21, 3, 5, 1.2, 0)
    mag = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
    return float(np.median(mag)), float(np.percentile(mag, 75)), float(mag.mean())


def draw_panel(cv2, np, label: str, frames: list, pair_indices: list[int]):
    canvas = []
    for frame_index, frame in zip(pair_indices, frames):
        out = frame.copy()
        cv2.putText(out, f"{label} f{frame_index}", (14, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        for name, (x1, y1, x2, y2) in ROIS.items():
            cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 255), 2)
            cv2.putText(out, name, (x1 + 4, y1 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
        canvas.append(out)
    return np.hstack(canvas)


def write_video(cv2, frames, path: Path, fps: float = 2.0):
    path.parent.mkdir(parents=True, exist_ok=True)
    h, w = frames[0].shape[:2]
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    if not writer.isOpened():
        raise RuntimeError(f"cannot create {path}")
    for frame in frames:
        writer.write(frame)
    writer.release()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("items", nargs="+", help="label=/path/to/root")
    args = parser.parse_args()

    cv2, np = import_cv2_np()
    roots = {}
    for item in args.items:
        label, raw = item.split("=", 1)
        roots[label] = Path(raw)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    frame_indices = [0, 3, 5, 8, 15, 25, 37, 58]
    pair_indices = [5, 8, 37, 58]
    rows = []
    panels = []
    for label, root in roots.items():
        frames = {idx: resize_to_audit(cv2, read_frame(cv2, root, idx)) for idx in frame_indices}
        panels.append(draw_panel(cv2, np, label, [frames[idx] for idx in pair_indices], pair_indices))
        for a, b in zip(frame_indices[:-1], frame_indices[1:]):
            for roi_name, roi in ROIS.items():
                med, p75, mean = flow_mag(cv2, np, frames[a], frames[b], roi)
                rows.append(
                    {
                        "candidate": label,
                        "pair": f"f{a}_f{b}",
                        "roi": roi_name,
                        "median_flow": f"{med:.4f}",
                        "p75_flow": f"{p75:.4f}",
                        "mean_flow": f"{mean:.4f}",
                    }
                )

    with (args.out_dir / "motion_roi_stats.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    contact = np.vstack(panels)
    cv2.imwrite(str(args.out_dir / "motion_roi_contact.jpg"), contact)
    write_video(cv2, panels, args.out_dir / "motion_roi_panels.mp4")


if __name__ == "__main__":
    main()

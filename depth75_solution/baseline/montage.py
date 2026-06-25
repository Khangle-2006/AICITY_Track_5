from __future__ import annotations

import math
from pathlib import Path

from .frame_io import numeric_pngs, read_rgb, write_rgb


def make_canvas(np, height: int, width: int, color=(245, 245, 245)):
    arr = np.zeros((height, width, 3), dtype=np.uint8)
    arr[:, :] = color
    return arr


def resize_keep_aspect(cv2, np, img, width: int, height: int):
    h, w = img.shape[:2]
    scale = min(width / w, height / h)
    nw = max(1, int(round(w * scale)))
    nh = max(1, int(round(h * scale)))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    canvas = make_canvas(np, height, width)
    x = (width - nw) // 2
    y = (height - nh) // 2
    canvas[y : y + nh, x : x + nw] = resized
    return canvas


def make_montage(cv2, np, row: dict[str, str], history_frames, output_dir: Path, montage_path: Path) -> None:
    generated_paths = numeric_pngs(output_dir)
    if not generated_paths:
        return

    sample_idxs = sorted({0, len(generated_paths) // 3, (2 * len(generated_paths)) // 3, len(generated_paths) - 1})
    generated = []
    for idx in sample_idxs:
        img = read_rgb(cv2, generated_paths[idx])
        if img is not None:
            generated.append((f"gen {idx}", img))

    hist_show = []
    if history_frames:
        hist_idxs = sorted({0, len(history_frames) // 2, len(history_frames) - 1})
        hist_show = [(f"hist {idx}", history_frames[idx]) for idx in hist_idxs]

    cells = hist_show + generated
    if not cells:
        return

    thumb_w, thumb_h = 220, 124
    gap = 8
    margin = 12
    label_h = 24
    header_h = 95
    cols = min(4, len(cells))
    rows_n = math.ceil(len(cells) / cols)
    width = margin * 2 + cols * thumb_w + (cols - 1) * gap
    height = header_h + rows_n * (label_h + thumb_h + gap) + margin
    canvas = make_canvas(np, height, width, color=(255, 255, 255))

    title = row.get("sample_id", "")
    meta = (
        f"part={row.get('dataset_part')} | view={row.get('view_type')} | "
        f"phase={row.get('phase')} | K={row.get('history_K')} | N={row.get('target_N')}"
    )
    cv2.putText(canvas, title[:120], (margin, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (20, 60, 130), 1, cv2.LINE_AA)
    cv2.putText(canvas, meta[:120], (margin, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (35, 35, 35), 1, cv2.LINE_AA)
    cv2.putText(canvas, str(output_dir)[-120:], (margin, 78), cv2.FONT_HERSHEY_SIMPLEX, 0.34, (80, 80, 80), 1, cv2.LINE_AA)

    for i, (label, img) in enumerate(cells):
        r = i // cols
        c = i % cols
        x = margin + c * (thumb_w + gap)
        y_label = header_h + r * (label_h + thumb_h + gap)
        y_img = y_label + label_h
        cv2.putText(canvas, label, (x, y_label + 17), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (20, 20, 20), 1, cv2.LINE_AA)
        thumb = resize_keep_aspect(cv2, np, img, thumb_w, thumb_h)
        canvas[y_img : y_img + thumb_h, x : x + thumb_w] = thumb
        cv2.rectangle(canvas, (x, y_img), (x + thumb_w - 1, y_img + thumb_h - 1), (200, 200, 200), 1)

    montage_path.parent.mkdir(parents=True, exist_ok=True)
    write_rgb(cv2, montage_path, canvas)

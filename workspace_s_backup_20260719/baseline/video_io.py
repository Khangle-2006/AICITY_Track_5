from __future__ import annotations

from pathlib import Path

from .naming import parse_int


def parse_float(value: str) -> float | None:
    try:
        if value == "":
            return None
        return float(value)
    except Exception:
        return None


def evenly_spaced_indices(n: int, count: int) -> list[int]:
    if n <= 0:
        return []
    if count <= 1:
        return [0]
    if n <= count:
        return list(range(n))
    return sorted({round(i * (n - 1) / (count - 1)) for i in range(count)})


def read_frame_at(cv2, video_path: Path, cap, index: int):
    """Read a frame robustly from videos whose random seeking is unreliable."""
    for _ in range(2):
        cap.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, bgr = cap.read()
        if ok and bgr is not None:
            return bgr, cap

        cap.release()
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            continue

    # Some BDD MP4 files fail on an exact seek but decode after seeking to an
    # earlier keyframe and reading forward.
    cap.release()
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None, cap
    start = max(0, index - 64)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    bgr = None
    for _ in range(start, index + 1):
        ok, frame = cap.read()
        if not ok or frame is None:
            return None, cap
        bgr = frame
    return bgr, cap


def extract_val_history_and_target(cv2, row: dict[str, str], num_history: int, target_gap_frames: int):
    video_path = Path(row.get("video_path", ""))
    if not video_path.exists():
        return [], None, {"error": f"missing video: {video_path}"}

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return [], None, {"error": f"cannot open video: {video_path}"}

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    duration = frame_count / fps if fps > 0 else 0.0

    start_time = parse_float(row.get("start_time", "")) or 0.0
    end_time = parse_float(row.get("end_time", ""))
    if end_time is None or end_time <= start_time:
        end_time = min(duration, start_time + 2.0)

    if frame_count <= 0:
        cap.release()
        return [], None, {"error": "video has zero frames"}

    start_frame = max(0, min(frame_count - 1, int(round(start_time * fps))))
    end_frame = max(start_frame, min(frame_count - 1, int(round(end_time * fps))))
    target_frame = end_frame
    history_end_frame = max(start_frame, target_frame - max(1, target_gap_frames))
    if history_end_frame <= start_frame and target_frame > start_frame:
        history_end_frame = target_frame - 1

    rel_idxs = evenly_spaced_indices(max(1, history_end_frame - start_frame + 1), num_history)
    history_indices = [start_frame + rel_idx for rel_idx in rel_idxs]
    frames = []
    for idx in history_indices:
        bgr, cap = read_frame_at(cv2, video_path, cap, idx)
        if bgr is not None:
            frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))

    target = None
    bgr, cap = read_frame_at(cv2, video_path, cap, target_frame)
    if bgr is not None:
        target = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    cap.release()

    meta = {
        "fps": fps,
        "start_frame": start_frame,
        "history_end_frame": history_end_frame,
        "history_indices": history_indices,
        "history_gap_frames": max(1, history_indices[-1] - history_indices[-2])
        if len(history_indices) >= 2
        else 1,
        "target_gap_frames": max(1, target_frame - history_end_frame),
        "target_frame": target_frame,
        "duration_frames": max(0, end_frame - start_frame + 1),
        "phase": parse_int(row.get("phase", ""), -1),
    }
    return frames, target, meta

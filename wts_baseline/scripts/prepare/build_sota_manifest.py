"""
SOTA Manifest Builder for AI City Challenge 2026 Track 5

Builds JSONL manifests with separate pedestrian/vehicle captions,
multi-view frame paths, and proper history/future frame selection
for the SOTA flow-matching pipeline.
"""

import json
import os
import argparse
from pathlib import Path


def get_frame_paths_for_scenario(scenario_dir, view_type="overhead"):
    """Get sorted frame paths for a scenario, preferring Camera1 or first available."""
    view_dir = scenario_dir / view_type

    if not view_dir.exists():
        return []

    mp4_files = sorted(view_dir.glob("*.mp4"))
    if not mp4_files:
        return []

    preferred = None
    for f in mp4_files:
        if "Camera1" in f.name or "_11_" in f.name:
            preferred = f
            break

    if preferred is None:
        preferred = mp4_files[0]

    meta_file = view_dir / "_meta.txt"
    fps = 30
    total_frames = 0

    if meta_file.exists():
        with open(meta_file, "r") as mf:
            for line in mf:
                if "fps" in line.lower():
                    try:
                        fps = float(line.split(":")[1].strip())
                    except (ValueError, IndexError):
                        pass
                if "total_frames" in line.lower():
                    try:
                        total_frames = int(line.split(":")[1].strip())
                    except (ValueError, IndexError):
                        pass

    frame_dir = scenario_dir
    frames = sorted(frame_dir.glob("*.png")) + sorted(frame_dir.glob("*.jpg"))

    if not frames:
        frames = sorted(view_dir.glob("*.png")) + sorted(view_dir.glob("*.jpg"))

    return frames, fps, total_frames


def time_to_frames(time_sec, fps=30):
    """Convert timestamp to frame index."""
    return int(time_sec * fps)


def select_history_frames(all_frames, history_start_frame, num_history, strategy="dense_tail"):
    """Select history frames with configurable strategy."""
    available = [f for f in all_frames if _frame_index(f) < history_start_frame]

    if not available:
        available = all_frames[:num_history]

    if len(available) <= num_history:
        return available

    if strategy == "tail":
        return available[-num_history:]
    elif strategy == "uniform":
        step = len(available) / num_history
        indices = [int(i * step) for i in range(num_history)]
        return [available[i] for i in indices]
    elif strategy == "dense_tail":
        tail_count = int(num_history * 0.75)
        head_count = num_history - tail_count
        tail = available[-tail_count:]
        head_step = max(1, len(available[:-tail_count]) // max(1, head_count))
        head = available[::head_step][:head_count]
        return head + tail
    else:
        return available[-num_history:]


def _frame_index(frame_path):
    """Extract frame index from path."""
    stem = Path(frame_path).stem
    try:
        return int(stem)
    except ValueError:
        parts = stem.split("_")
        try:
            return int(parts[-1])
        except ValueError:
            return 0


def select_future_frames(all_frames, future_start_frame, num_future):
    """Select future frames starting from the phase start."""
    future = [f for f in all_frames if _frame_index(f) >= future_start_frame]
    return future[:num_future]


def build_manifest_from_parsed(parsed_path, frame_root, output_path,
                                num_history=16, num_future=8,
                                history_strategy="dense_tail",
                                fps=30):
    """Build JSONL manifest from parsed caption JSON."""
    with open(parsed_path, "r") as f:
        entries = json.load(f)

    manifest_lines = []
    skipped = 0

    for entry in entries:
        scenario_name = entry["scenario_name"]
        split = entry["split"]
        view_type = entry.get("view_type", "overhead")

        scenario_dir = Path(frame_root) / split / scenario_name

        if not scenario_dir.exists():
            skipped += 1
            continue

        frames, _, _ = get_frame_paths_for_scenario(scenario_dir, view_type)

        if len(frames) < num_history + num_future:
            skipped += 1
            continue

        history_start_frame = time_to_frames(entry["start_time"], fps) - num_history
        future_start_frame = time_to_frames(entry["start_time"], fps)

        history_paths = select_history_frames(frames, history_start_frame, num_history, history_strategy)
        future_paths = select_future_frames(frames, future_start_frame, num_future)

        if len(history_paths) < num_history or len(future_paths) < num_future:
            skipped += 1
            continue

        history_strs = [str(p) for p in history_paths]
        future_strs = [str(p) for p in future_paths]

        if not all(os.path.exists(p) for p in history_strs + future_strs):
            skipped += 1
            continue

        manifest_entry = {
            "history_frames": history_strs,
            "future_frames": future_strs,
            "caption_pedestrian": entry["caption_pedestrian"],
            "caption_vehicle": entry["caption_vehicle"],
            "merged_caption": f"{entry['caption_vehicle']} {entry['caption_pedestrian']}",
            "phase_label": entry["phase_label"],
            "scenario_name": scenario_name,
            "view_type": view_type,
        }
        manifest_lines.append(manifest_entry)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w") as f:
        for line in manifest_lines:
            f.write(json.dumps(line) + "\n")

    print(f"Built manifest: {len(manifest_lines)} samples (skipped {skipped}) -> {output_path}")
    return manifest_lines


def build_test_manifest_sota(test_root, output_path, num_history=16, fps=30):
    """Build test manifest from WTS_TRACK5_TEST directory structure."""
    test_dir = Path(test_root)
    if not test_dir.exists():
        print(f"Test directory {test_dir} does not exist.")
        return []

    manifest_lines = []

    for sample_dir in sorted(test_dir.iterdir()):
        if not sample_dir.is_dir():
            continue

        caption_file = sample_dir / "caption.json"
        input_dir = sample_dir / "input"

        if not caption_file.exists() or not input_dir.exists():
            continue

        with open(caption_file, "r") as f:
            caption_data = json.load(f)

        input_frames = sorted(input_dir.glob("*.png")) + sorted(input_dir.glob("*.jpg"))
        input_strs = [str(p) for p in input_frames]

        if len(input_strs) < num_history:
            input_strs = input_strs + [input_strs[-1]] * (num_history - len(input_strs))
        elif len(input_strs) > num_history:
            input_strs = input_strs[-num_history:]

        phases = caption_data.get("event_phase", [])
        if not phases:
            continue

        phase = phases[0]
        frame_length = caption_data.get("frame length", 68)

        manifest_entry = {
            "history_frames": input_strs,
            "caption_pedestrian": phase.get("caption_pedestrian", ""),
            "caption_vehicle": phase.get("caption_vehicle", ""),
            "merged_caption": f"{phase.get('caption_vehicle', '')} {phase.get('caption_pedestrian', '')}",
            "phase_label": int(phase.get("labels", [0])[0]) if phase.get("labels") else 0,
            "scenario_name": sample_dir.name,
            "frame_length": frame_length,
            "view_type": "test",
        }
        manifest_lines.append(manifest_entry)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w") as f:
        for line in manifest_lines:
            f.write(json.dumps(line) + "\n")

    print(f"Built test manifest: {len(manifest_lines)} samples -> {output_path}")
    return manifest_lines


def main():
    parser = argparse.ArgumentParser(description="Build SOTA manifest with dual captions")
    parser.add_argument("--parsed_dir", type=str, default="data/parsed",
                        help="Directory containing parsed JSON files")
    parser.add_argument("--frame_root", type=str, default="data/frames",
                        help="Root directory containing extracted frames")
    parser.add_argument("--output_dir", type=str, default="data/manifests",
                        help="Output directory for manifest files")
    parser.add_argument("--num_history", type=int, default=16,
                        help="Number of history frames")
    parser.add_argument("--num_future", type=int, default=8,
                        help="Number of future frames to predict")
    parser.add_argument("--history_strategy", type=str, default="dense_tail",
                        choices=["tail", "uniform", "dense_tail"])
    parser.add_argument("--fps", type=int, default=30,
                        help="Frames per second for time-to-frame conversion")
    parser.add_argument("--build_test", action="store_true",
                        help="Build test manifest instead of train/val")
    parser.add_argument("--test_root", type=str,
                        default="/data/Track_5/WTS_dataset/WTS_dataset/WTS_TRACK5_TEST/WTS_TRACK5_TEST",
                        help="Test directory root")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    if args.build_test:
        build_test_manifest_sota(args.test_root,
                                  os.path.join(args.output_dir, "test_sota.jsonl"),
                                  args.num_history, args.fps)
    else:
        for split in ["train", "val"]:
            parsed_path = os.path.join(args.parsed_dir, f"{split}_parsed_sota.json")
            if not os.path.exists(parsed_path):
                print(f"Warning: {parsed_path} not found, skipping.")
                continue

            output_path = os.path.join(args.output_dir, f"{split}_sota.jsonl")
            build_manifest_from_parsed(
                parsed_path, args.frame_root, output_path,
                args.num_history, args.num_future,
                args.history_strategy, args.fps
            )


if __name__ == "__main__":
    main()

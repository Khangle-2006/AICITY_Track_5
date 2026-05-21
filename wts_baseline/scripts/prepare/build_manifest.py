"""
Builds train.jsonl and val.jsonl manifests from parsed annotations + extracted frames.

Each JSONL line contains:
  {
    "history_frames": ["path/000010.jpg", ...],  # H paths
    "future_frames": ["path/000018.jpg", ...],    # T paths
    "merged_caption": "...",
    "phase_label": 3
  }
"""

import json
import argparse
from pathlib import Path


def build_manifest(parsed_json_path, frames_dir, output_path, h=8, t=8, fps=30.0):
    with open(parsed_json_path, 'r') as f:
        data = json.load(f)

    frames_dir = Path(frames_dir)
    valid_samples = []
    skipped = {"no_frames_dir": 0, "too_short": 0, "no_history": 0, "missing_files": 0}

    for item in data:
        scenario_name = item['scenario_name']
        split = item['split']
        scenario_frames_dir = frames_dir / split / scenario_name

        if not scenario_frames_dir.exists():
            skipped["no_frames_dir"] += 1
            continue

        start_time = item['start_time']
        end_time = item['end_time']

        start_frame = int(start_time * fps)
        end_frame = int(end_time * fps)

        # Future frames come from the phase itself
        phase_length = end_frame - start_frame
        if phase_length < t:
            skipped["too_short"] += 1
            continue

        # History frames come from right before start_frame
        if start_frame < h:
            skipped["no_history"] += 1
            continue

        # Sample H history frames (stride=1 for simplicity)
        history_indices = list(range(start_frame - h, start_frame))
        # Sample T future frames from the beginning of the phase
        future_indices = list(range(start_frame, start_frame + t))

        history_paths = [str(scenario_frames_dir / f"{idx:06d}.jpg") for idx in history_indices]
        future_paths = [str(scenario_frames_dir / f"{idx:06d}.jpg") for idx in future_indices]

        # Check all files exist
        all_paths = history_paths + future_paths
        if not all(Path(p).exists() for p in all_paths):
            skipped["missing_files"] += 1
            continue

        valid_samples.append({
            "history_frames": history_paths,
            "future_frames": future_paths,
            "merged_caption": item['merged_caption'],
            "phase_label": item['phase_label']
        })

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        for s in valid_samples:
            f.write(json.dumps(s) + "\n")

    print(f"Valid samples: {len(valid_samples)}")
    print(f"Skipped: {skipped}")
    print(f"Saved to {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--parsed_json", type=str, required=True,
                        help="Output from prepare_wts.py")
    parser.add_argument("--frames_dir", type=str,
                        default="/workspace/wts_baseline/data/frames",
                        help="Root directory with extracted frames")
    parser.add_argument("--output", type=str, required=True,
                        help="Output JSONL manifest path")
    parser.add_argument("--h", type=int, default=8)
    parser.add_argument("--t", type=int, default=8)
    args = parser.parse_args()

    build_manifest(args.parsed_json, args.frames_dir, args.output, h=args.h, t=args.t)


if __name__ == "__main__":
    main()

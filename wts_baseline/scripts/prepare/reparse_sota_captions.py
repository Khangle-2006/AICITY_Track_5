"""
Re-parse WTS captions from raw JSON files, preserving dual caption separation.
"""

import json
import os
import argparse
from pathlib import Path
from collections import defaultdict


def parse_raw_caption(caption_path):
    """Parse raw WTS caption JSON with dual captions."""
    with open(caption_path, "r") as f:
        data = json.load(f)

    phases = []
    for phase in data.get("event_phase", []):
        labels = phase.get("labels", [])
        phase_label = int(labels[0]) if labels else 0

        phases.append({
            "start_time": float(phase.get("start_time", 0)),
            "end_time": float(phase.get("end_time", 0)),
            "caption_pedestrian": phase.get("caption_pedestrian", ""),
            "caption_vehicle": phase.get("caption_vehicle", ""),
            "phase_label": phase_label,
        })

    return phases


def process_split(split, data_root, output_dir):
    """Process a split and output parsed JSON with dual captions."""
    caption_root = Path(data_root) / "caption" / split

    if not caption_root.exists():
        print(f"Warning: {caption_root} does not exist.")
        return []

    all_entries = []

    for scenario_dir in sorted(caption_root.iterdir()):
        if not scenario_dir.is_dir():
            continue

        scenario_name = scenario_dir.name

        for view_dir in sorted(scenario_dir.iterdir()):
            if not view_dir.is_dir():
                continue

            view_type = view_dir.name
            caption_file = view_dir / f"{scenario_name}_caption.json"

            if not caption_file.exists():
                continue

            phases = parse_raw_caption(str(caption_file))

            for phase_idx, phase in enumerate(phases):
                entry = {
                    "scenario_name": scenario_name,
                    "split": split,
                    "view_type": view_type,
                    "start_time": phase["start_time"],
                    "end_time": phase["end_time"],
                    "caption_pedestrian": phase["caption_pedestrian"],
                    "caption_vehicle": phase["caption_vehicle"],
                    "phase_label": phase["phase_label"],
                    "original_id": f"{scenario_name}_{view_type}_phase{phase_idx}",
                }
                all_entries.append(entry)

    output_path = Path(output_dir) / f"{split}_parsed_sota.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(all_entries, f, indent=2)

    print(f"Parsed {len(all_entries)} phases from {split} -> {output_path}")
    return all_entries


def main():
    parser = argparse.ArgumentParser(description="Re-parse WTS captions with dual separation")
    parser.add_argument("--data_root", type=str,
                        default="/data/Track_5/WTS_dataset/WTS_dataset")
    parser.add_argument("--output_dir", type=str, default="data/parsed")
    parser.add_argument("--splits", type=str, nargs="+", default=["train", "val"])
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    for split in args.splits:
        process_split(split, args.data_root, args.output_dir)


if __name__ == "__main__":
    main()

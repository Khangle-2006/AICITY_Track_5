"""
SOTA Data Preparation for AI City Challenge 2026 Track 5

Parses WTS annotations with dual caption separation (pedestrian/vehicle),
handles multi-view synchronization, and prepares structured JSON output
for the SOTA pipeline training.
"""

import json
import os
import argparse
from pathlib import Path


PHASE_LABEL_MAP = {
    "0": 0,
    "1": 1,
    "2": 2,
    "3": 3,
    "4": 4,
}


def parse_caption_file(caption_path):
    """Parse a single WTS caption JSON file."""
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

    return {
        "id": data.get("id", ""),
        "phases": phases,
    }


def parse_external_caption_file(caption_path):
    """Parse BDD_PC_5K external caption format."""
    with open(caption_path, "r") as f:
        data = json.load(f)

    phases = []
    if isinstance(data, list):
        entries = data
    else:
        entries = [data]

    for entry in entries:
        phases.append({
            "start_time": float(entry.get("start_time", 0)),
            "end_time": float(entry.get("end_time", 0)),
            "caption_pedestrian": entry.get("caption_pedestrian", entry.get("caption", "")),
            "caption_vehicle": entry.get("caption_vehicle", entry.get("caption", "")),
            "phase_label": int(entry.get("phase_label", entry.get("labels", [0])[0])),
        })

    return {"id": data[0].get("id", "") if isinstance(data, list) else data.get("id", ""), "phases": phases}


def process_split(split, data_root, output_dir, view_type="overhead"):
    """Process a full split (train/val) and save parsed JSON."""
    video_root = Path(data_root) / "video" / split
    caption_root = Path(data_root) / "caption" / split

    if not video_root.exists():
        print(f"Warning: {video_root} does not exist, skipping.")
        return []

    all_entries = []

    for scenario_dir in sorted(video_root.iterdir()):
        if not scenario_dir.is_dir():
            continue

        scenario_name = scenario_dir.name
        view_dir = scenario_dir / view_type

        if not view_dir.exists():
            continue

        caption_dir = caption_root / scenario_name / view_type
        caption_file = caption_dir / f"{scenario_name}_caption.json"

        if not caption_file.exists():
            continue

        parsed = parse_caption_file(str(caption_file))

        for phase_idx, phase in enumerate(parsed["phases"]):
            entry = {
                "scenario_name": scenario_name,
                "split": split,
                "view_type": view_type,
                "start_time": phase["start_time"],
                "end_time": phase["end_time"],
                "caption_pedestrian": phase["caption_pedestrian"],
                "caption_vehicle": phase["caption_vehicle"],
                "phase_label": phase["phase_label"],
                "original_id": f"{scenario_name}_phase{phase_idx}",
            }
            all_entries.append(entry)

    output_path = Path(output_dir) / f"{split}_parsed_sota.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(all_entries, f, indent=2)

    print(f"Parsed {len(all_entries)} phases from {split} split -> {output_path}")
    return all_entries


def process_external_split(split, data_root, output_dir):
    """Process BDD_PC_5K external dataset."""
    video_root = Path(data_root) / "external" / "video" / split
    caption_root = Path(data_root) / "external" / "caption" / split

    if not video_root.exists():
        print(f"Warning: External {video_root} does not exist, skipping.")
        return []

    all_entries = []
    video_files = list(video_root.glob("*.mp4"))

    for video_path in sorted(video_files):
        video_name = video_path.stem
        caption_file = caption_root / f"{video_name}_caption.json"

        if not caption_file.exists():
            caption_file = caption_root / f"{video_name}.json"

        if not caption_file.exists():
            continue

        try:
            parsed = parse_external_caption_file(str(caption_file))
        except Exception as e:
            print(f"Error parsing {caption_file}: {e}")
            continue

        for phase_idx, phase in enumerate(parsed["phases"]):
            entry = {
                "scenario_name": f"bdd_{video_name}",
                "split": f"external_{split}",
                "view_type": "front",
                "start_time": phase["start_time"],
                "end_time": phase["end_time"],
                "caption_pedestrian": phase["caption_pedestrian"],
                "caption_vehicle": phase["caption_vehicle"],
                "phase_label": phase["phase_label"],
                "original_id": f"bdd_{video_name}_phase{phase_idx}",
                "video_path": str(video_path),
            }
            all_entries.append(entry)

    output_path = Path(output_dir) / f"external_{split}_parsed.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(all_entries, f, indent=2)

    print(f"Parsed {len(all_entries)} phases from external {split} -> {output_path}")
    return all_entries


def main():
    parser = argparse.ArgumentParser(description="Parse WTS captions with dual caption separation")
    parser.add_argument("--data_root", type=str, default="/data/Track_5/WTS_dataset/WTS_dataset",
                        help="Root directory of WTS dataset")
    parser.add_argument("--output_dir", type=str, default="data/parsed",
                        help="Output directory for parsed JSON files")
    parser.add_argument("--splits", type=str, nargs="+", default=["train", "val"],
                        help="Splits to process")
    parser.add_argument("--view_type", type=str, default="overhead",
                        choices=["overhead", "vehicle"],
                        help="Camera view type to process")
    parser.add_argument("--include_external", action="store_true",
                        help="Include BDD_PC_5K external dataset")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    for split in args.splits:
        process_split(split, args.data_root, args.output_dir, args.view_type)

    if args.include_external:
        for split in args.splits:
            process_external_split(split, args.data_root, args.output_dir)


if __name__ == "__main__":
    main()

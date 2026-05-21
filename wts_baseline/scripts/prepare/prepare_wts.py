"""
Parses WTS annotation JSONs from the real dataset structure.

Dataset structure:
  caption/{train,val}/<scenario>/overhead_view/<scenario>_caption.json

Each JSON has:
  {
    "id": 726,
    "overhead_videos": ["Camera1_0.mp4", ...],
    "event_phase": [
      {
        "labels": ["4"],
        "caption_pedestrian": "...",
        "caption_vehicle": "...",
        "start_time": "37.734",
        "end_time": "42.275"
      }, ...
    ]
  }
"""

import json
import argparse
from pathlib import Path


def parse_wts_annotations(data_root, split, output_file):
    caption_dir = Path(data_root) / "caption" / split
    if not caption_dir.exists():
        print(f"[ERROR] {caption_dir} does not exist")
        return

    scenarios = sorted([d for d in caption_dir.iterdir() if d.is_dir()])
    parsed_data = []

    for scenario_dir in scenarios:
        scenario_name = scenario_dir.name
        overhead_caption_dir = scenario_dir / "overhead_view"
        if not overhead_caption_dir.exists():
            continue

        json_files = list(overhead_caption_dir.glob("*_caption.json"))
        if not json_files:
            continue

        for jf in json_files:
            with open(jf, 'r') as f:
                data = json.load(f)

            video_id = data.get("id")

            for phase in data.get("event_phase", []):
                start_time = float(phase.get("start_time", 0))
                end_time = float(phase.get("end_time", 0))
                cap_ped = phase.get("caption_pedestrian", "")
                cap_veh = phase.get("caption_vehicle", "")
                labels = phase.get("labels", [])

                merged_caption = f"{cap_veh} {cap_ped}".strip()
                phase_label = int(labels[0]) if labels else 0

                parsed_data.append({
                    "scenario_name": scenario_name,
                    "split": split,
                    "start_time": start_time,
                    "end_time": end_time,
                    "merged_caption": merged_caption,
                    "phase_label": phase_label,
                    "original_id": video_id
                })

    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(parsed_data, f, indent=2)

    print(f"Parsed {len(parsed_data)} event phases from {len(scenarios)} scenarios ({split}).")
    print(f"Saved to {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Parse WTS caption JSONs")
    parser.add_argument("--data_root", type=str,
                        default="/data/Track_5/WTS_dataset/WTS_dataset")
    parser.add_argument("--split", type=str, default="train",
                        choices=["train", "val"])
    parser.add_argument("--output_file", type=str,
                        default="/workspace/wts_baseline/data/parsed/train_parsed.json")
    args = parser.parse_args()

    parse_wts_annotations(args.data_root, args.split, args.output_file)


if __name__ == "__main__":
    main()

"""
Merge existing manifests with parsed dual caption data.
"""

import json
import os
import argparse
from pathlib import Path
from collections import defaultdict


def load_parsed_captions(parsed_path):
    """Load parsed captions indexed by scenario_name + phase_label."""
    with open(parsed_path, "r") as f:
        entries = json.load(f)

    index = defaultdict(list)
    for entry in entries:
        key = (entry["scenario_name"], entry["phase_label"])
        index[key].append(entry)

    return index


def extract_scenario(frame_path):
    """Extract scenario name from frame path."""
    return Path(frame_path).parent.name


def merge_manifests(manifest_path, parsed_path, output_path):
    """Merge manifest with parsed captions."""
    captions = load_parsed_captions(parsed_path)

    with open(manifest_path, "r") as f:
        samples = [json.loads(line) for line in f if line.strip()]

    merged = []
    matched = 0
    unmatched = 0

    for sample in samples:
        scenario = extract_scenario(sample["history_frames"][0])
        phase = sample.get("phase_label", 0)

        key = (scenario, phase)

        if key in captions:
            cap_entry = captions[key][0]
            sample["caption_pedestrian"] = cap_entry.get("caption_pedestrian", "")
            sample["caption_vehicle"] = cap_entry.get("caption_vehicle", "")
            sample["scenario_name"] = scenario
            matched += 1
        else:
            mc = sample.get("merged_caption", "")
            half = len(mc) // 2
            sample["caption_pedestrian"] = mc[half:]
            sample["caption_vehicle"] = mc[:half]
            sample["scenario_name"] = scenario
            unmatched += 1

        merged.append(sample)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w") as f:
        for sample in merged:
            f.write(json.dumps(sample) + "\n")

    print(f"Merged {len(merged)} samples: {matched} matched, {unmatched} fallback")
    print(f"Output: {output_path}")
    return merged


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest_dir", default="data/manifests")
    parser.add_argument("--parsed_dir", default="data/parsed")
    parser.add_argument("--output_dir", default="data/manifests")
    args = parser.parse_args()

    for split in ["train", "val"]:
        manifest = os.path.join(args.manifest_dir, f"{split}.jsonl")
        parsed = os.path.join(args.parsed_dir, f"{split}_parsed_sota.json")
        output = os.path.join(args.output_dir, f"{split}_sota.jsonl")

        if not os.path.exists(manifest):
            print(f"Skipping {split}: manifest not found")
            continue
        if not os.path.exists(parsed):
            print(f"Skipping {split}: parsed not found")
            continue

        merge_manifests(manifest, parsed, output)


if __name__ == "__main__":
    main()

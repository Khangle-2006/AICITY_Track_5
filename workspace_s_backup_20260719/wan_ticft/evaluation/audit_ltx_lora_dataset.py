import argparse
import json
from collections import Counter
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--check_files", action="store_true")
    args = parser.parse_args()

    manifest = Path(args.manifest)
    rows = []
    with manifest.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))

    views = Counter()
    phases = Counter()
    clip_lengths = Counter()
    missing = 0
    bad_layout = 0
    empty_prompts = 0

    for row in rows:
        condition = list(row.get("frames", {}).get("condition", []))
        target = list(row.get("frames", {}).get("target", []))
        views[row.get("view_type", "unknown")] += 1
        phases[str(row.get("phase", "unknown"))] += 1

        # Use the final observed frame as frame 0, followed by 120 future frames.
        clip = condition[-1:] + target
        clip_lengths[len(clip)] += 1
        if len(clip) % 8 != 1:
            bad_layout += 1
        if not str(row.get("prompt", "")).strip():
            empty_prompts += 1
        if args.check_files:
            missing += sum(not Path(path).is_file() for path in clip)

    print("=== LTX LoRA dataset audit ===")
    print(f"manifest: {manifest}")
    print(f"rows: {len(rows)}")
    print(f"views: {dict(views)}")
    print(f"phases: {dict(phases)}")
    print(f"proposed clip lengths: {dict(clip_lengths)}")
    print(f"clips violating 8n+1: {bad_layout}")
    print(f"empty prompts: {empty_prompts}")
    if args.check_files:
        print(f"missing proposed clip frames: {missing}")
    else:
        print("file existence check: skipped")

    if not rows or bad_layout or empty_prompts or missing:
        raise SystemExit(1)

    print("gate: PASS")
    print("proposed clip contract: final condition frame + 120 target frames = 121 frames")


if __name__ == "__main__":
    main()

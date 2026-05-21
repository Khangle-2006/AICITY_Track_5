"""
Frame extractor for WTS dataset.
Extracts frames from overhead_view videos for each scenario.

Dataset structure:
  /data/Track_5/WTS_dataset/WTS_dataset/video/{train,val}/<scenario>/overhead_view/*.mp4
"""

import os
import cv2
import argparse
from pathlib import Path
from tqdm import tqdm

def extract_frames_from_video(video_path, output_dir):
    """Extracts all frames from a single video file."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"  [WARN] Cannot open: {video_path}")
        return 0

    fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    os.makedirs(output_dir, exist_ok=True)

    count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        out_path = os.path.join(output_dir, f"{count:06d}.jpg")
        cv2.imwrite(out_path, frame)
        count += 1

    cap.release()
    return count


def main():
    parser = argparse.ArgumentParser(description="Extract frames from WTS overhead videos.")
    parser.add_argument("--data_root", type=str,
                        default="/data/Track_5/WTS_dataset/WTS_dataset",
                        help="Root of WTS dataset")
    parser.add_argument("--output_root", type=str,
                        default="/workspace/wts_baseline/data/frames",
                        help="Where to save extracted frames")
    parser.add_argument("--splits", type=str, nargs="+", default=["train", "val"],
                        help="Which splits to process")
    args = parser.parse_args()

    data_root = Path(args.data_root)

    for split in args.splits:
        video_split_dir = data_root / "video" / split
        if not video_split_dir.exists():
            print(f"[SKIP] {video_split_dir} does not exist")
            continue

        scenarios = sorted([d for d in video_split_dir.iterdir() if d.is_dir()])
        print(f"\n=== Processing {split}: {len(scenarios)} scenarios ===")

        for scenario_dir in tqdm(scenarios, desc=f"{split}"):
            scenario_name = scenario_dir.name
            overhead_dir = scenario_dir / "overhead_view"
            if not overhead_dir.exists():
                continue

            # Use the first camera (Camera1) as the primary overhead view
            mp4_files = sorted(overhead_dir.glob("*.mp4"))
            if not mp4_files:
                continue

            # Pick Camera1 if available, otherwise pick the first video
            chosen = None
            for f in mp4_files:
                if "Camera1" in f.name:
                    chosen = f
                    break
            if chosen is None:
                chosen = mp4_files[0]

            out_dir = os.path.join(args.output_root, split, scenario_name)
            if os.path.exists(out_dir) and len(os.listdir(out_dir)) > 0:
                continue  # Skip already extracted

            n = extract_frames_from_video(chosen, out_dir)
            # Save a small metadata file
            meta_path = os.path.join(out_dir, "_meta.txt")
            cap = cv2.VideoCapture(str(chosen))
            fps = cap.get(cv2.CAP_PROP_FPS)
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap.release()
            with open(meta_path, 'w') as f:
                f.write(f"video={chosen.name}\nfps={fps}\nwidth={w}\nheight={h}\ntotal_frames={n}\n")

    print("\nFrame extraction complete.")


if __name__ == "__main__":
    main()

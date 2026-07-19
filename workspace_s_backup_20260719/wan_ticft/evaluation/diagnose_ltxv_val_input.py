from __future__ import annotations

import csv
from pathlib import Path

import imageio.v2 as imageio


def write_frames(video_path: Path, output_dir: Path, indices: list[int]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    reader = imageio.get_reader(video_path)
    for index in indices:
        imageio.imwrite(output_dir / f"frame_{index:03d}.png", reader.get_data(index))
    reader.close()


def main() -> None:
    root = Path("/workspace/experiments/wan_ticft/ltxv_bdd_val2_gate")
    output = Path("/workspace/experiments/wan_ticft/ltxv_bdd_val2_diagnostic")
    with (root / "report.csv").open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    for row in rows:
        sample_id = row["sample_id"]
        history_frames = int(row["history_frames"])
        history = root / "history_videos" / f"{sample_id}.mp4"
        raw = next((root / "raw" / sample_id).glob("*.mp4"))
        write_frames(history, output / sample_id / "history", [0, history_frames - 1])
        write_frames(raw, output / sample_id / "raw", [0, history_frames - 1, history_frames])
    print(output)


if __name__ == "__main__":
    main()

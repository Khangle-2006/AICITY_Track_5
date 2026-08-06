from __future__ import annotations

import random
from pathlib import Path

from .index_io import read_csv
from .refiner_features import make_refiner_arrays
from .video_io import extract_val_history_and_target


def filter_rows(rows: list[dict[str, str]], dataset_part: str = "", view_type: str = "", phase: str = ""):
    out = rows
    if dataset_part:
        out = [r for r in out if r.get("dataset_part", "") == dataset_part]
    if view_type:
        out = [r for r in out if r.get("view_type", "") == view_type]
    if phase:
        out = [r for r in out if str(r.get("phase", "")) == str(phase)]
    return out


def load_split_rows(index_dir: Path, split: str) -> list[dict[str, str]]:
    path = index_dir / f"{split}_index.csv"
    if not path.exists():
        raise SystemExit(f"Missing index CSV: {path}")
    return read_csv(path)


def sample_rows(rows: list[dict[str, str]], max_rows: int, seed: int) -> list[dict[str, str]]:
    rows = sorted(rows, key=lambda r: r.get("sample_id", ""))
    if max_rows <= 0 or len(rows) <= max_rows:
        return rows
    rng = random.Random(seed)
    return rng.sample(rows, max_rows)


class RefinerDataset:
    """Lazy OpenCV dataset for trainable residual refiner."""

    def __init__(self, cv2, np, torch, rows, args, feature_cfg, train: bool):
        self.cv2 = cv2
        self.np = np
        self.torch = torch
        self.rows = rows
        self.args = args
        self.feature_cfg = feature_cfg
        self.train = train
        self.rng = random.Random(args.seed + (101 if train else 202))

    def __len__(self):
        return len(self.rows)

    def _resize(self, img):
        return self.cv2.resize(img, (self.args.train_width, self.args.train_height), interpolation=self.cv2.INTER_AREA)

    def _get_item(self, idx: int):
        row = self.rows[idx]
        target_gap = self.args.target_gap_frames
        if self.train and getattr(self.args, "random_horizon", False):
            target_gap = self.rng.randint(self.args.horizon_min, self.args.horizon_max)
        frames, target, meta = extract_val_history_and_target(
            self.cv2,
            row,
            self.args.num_history,
            target_gap,
        )
        if meta and "error" in meta:
            raise RuntimeError(meta["error"])
        if not frames or target is None:
            raise RuntimeError(f"missing frames/target for {row.get('sample_id', '')}")

        frames = [self._resize(f) for f in frames]
        target = self._resize(target)
        horizon = max(1, int(meta.get("target_gap_frames", target_gap)))
        if not self.train:
            horizon = max(1, self.args.eval_horizon)
        history_gap = max(1, int(meta.get("history_gap_frames", 1)))

        features, target, base = make_refiner_arrays(
            self.cv2,
            self.np,
            frames,
            target,
            row.get("view_type", ""),
            horizon,
            self.feature_cfg,
            history_gap,
            row.get("phase", ""),
        )

        x = self.torch.from_numpy(features.astype(self.np.float32) / 255.0).permute(2, 0, 1)
        y = self.torch.from_numpy(target.astype(self.np.float32) / 255.0).permute(2, 0, 1)
        b = self.torch.from_numpy(base.astype(self.np.float32) / 255.0).permute(2, 0, 1)
        return {
            "x": x,
            "target": y,
            "base": b,
            "sample_id": row.get("sample_id", ""),
            "view_type": row.get("view_type", ""),
            "dataset_part": row.get("dataset_part", ""),
            "phase": row.get("phase", ""),
        }

    def __getitem__(self, idx: int):
        try:
            return self._get_item(idx)
        except (RuntimeError, OSError, ValueError):
            # A small number of source MP4s remain unreadable even after robust
            # seek retries. Skip only those rows instead of aborting an epoch.
            return None

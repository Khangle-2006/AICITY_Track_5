import json

import torch
from PIL import Image
from torch.utils.data import Dataset

from src.datasets.transforms import get_train_transforms, get_val_transforms


def infer_sample_type(scenario_name):
    """Classify a sample from its manifest name for domain-aware conditioning."""
    name = str(scenario_name)
    if name.startswith("video"):
        return "bdd_front"
    if "Camera" in name:
        return "wts_overhead"
    return "wts_vehicle"


def select_frame_paths(paths, max_frames=None, strategy="tail"):
    """Select a bounded history window while preserving temporal order."""
    paths = list(paths)
    if max_frames is None or max_frames <= 0 or len(paths) <= max_frames:
        return paths

    if strategy == "uniform":
        if max_frames == 1:
            return [paths[-1]]
        indices = torch.linspace(0, len(paths) - 1, steps=max_frames).round().long().tolist()
        return [paths[i] for i in indices]

    if strategy == "dense_tail":
        tail_count = max(1, int(max_frames * 0.75))
        head_count = max_frames - tail_count
        tail = paths[-tail_count:]
        if head_count <= 0:
            return tail
        head_pool = paths[: -tail_count]
        if not head_pool:
            return tail
        if head_count == 1:
            head = [head_pool[-1]]
        else:
            indices = torch.linspace(0, len(head_pool) - 1, steps=head_count).round().long().tolist()
            head = [head_pool[i] for i in indices]
        return head + tail

    return paths[-max_frames:]


class WTSDataset(Dataset):
    def __init__(self, manifest_path, is_train=True, size=(288, 512),
                 max_history_frames=None, history_strategy="tail",
                 use_window_shift=False, history_len=None, future_len=None):
        """
        manifest_path: Path to train.jsonl, val.jsonl, or test.jsonl.
        is_train: Boolean to toggle training transforms.
        size: Target (height, width) for frames.
        max_history_frames: Optional cap for variable-K conditioning.
        history_strategy: tail, dense_tail, or uniform.
        use_window_shift: Whether to randomly shift the history/future boundary during training.
        history_len: Target history length for shifting (defaults to len(history_frames)).
        future_len: Target future length for shifting (defaults to len(future_frames)).
        """
        super().__init__()
        self.is_train = is_train
        self.max_history_frames = max_history_frames
        self.history_strategy = history_strategy
        self.use_window_shift = use_window_shift
        self.history_len = history_len
        self.future_len = future_len
        self.samples = []
        with open(manifest_path, "r") as f:
            for line in f:
                if line.strip():
                    self.samples.append(json.loads(line))

        self.transforms = get_train_transforms(size) if is_train else get_val_transforms(size)

    def __len__(self):
        return len(self.samples)

    def _load_frames(self, paths):
        frames = []
        for p in paths:
            img = Image.open(p).convert("RGB")
            tensor = self.transforms(img)
            frames.append(tensor)
        return torch.stack(frames, dim=0)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        scenario_name = sample.get("scenario_name", f"sample_{idx}")
        
        history_paths = sample["history_frames"]
        future_paths = sample.get("future_frames", [])
        
        if self.is_train and self.use_window_shift and len(future_paths) > 0:
            import random
            all_paths = history_paths + future_paths
            L_total = len(all_paths)
            
            h_len = self.history_len or self.max_history_frames or len(history_paths)
            f_len = self.future_len or len(future_paths)
            segment_len = h_len + f_len
            
            if L_total >= segment_len:
                start_idx = random.randint(0, L_total - segment_len)
                history_paths = all_paths[start_idx : start_idx + h_len]
                future_paths = all_paths[start_idx + h_len : start_idx + h_len + f_len]
            else:
                history_paths = select_frame_paths(
                    sample["history_frames"],
                    max_frames=self.max_history_frames,
                    strategy=self.history_strategy,
                )
        else:
            history_paths = select_frame_paths(
                sample["history_frames"],
                max_frames=self.max_history_frames,
                strategy=self.history_strategy,
            )

        history_tensor = self._load_frames(history_paths)

        if self.is_train and torch.rand(1).item() < 0.5:
            history_tensor = torch.flip(history_tensor, dims=[-1])

        res = {
            "history_frames": history_tensor,
            "merged_caption": sample["merged_caption"],
            "phase_label": torch.tensor(sample["phase_label"], dtype=torch.long),
            "scenario_name": scenario_name,
            "sample_type": sample.get("sample_type", infer_sample_type(scenario_name)),
            "history_paths": history_paths,
            "original_history_length": len(sample["history_frames"]),
        }
        if "frame_length" in sample:
            res["frame_length"] = sample["frame_length"]

        if len(future_paths) > 0:
            future_tensor = self._load_frames(future_paths)
            if self.is_train and torch.rand(1).item() < 0.5:
                future_tensor = torch.flip(future_tensor, dims=[-1])
            res["future_frames"] = future_tensor

        return res

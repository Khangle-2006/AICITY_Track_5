"""
SOTA Dataset for AI City Challenge 2026 Track 5

Extends WTSDataset with dual caption support (pedestrian/vehicle separation),
multi-view awareness, and variable-length future frame handling for the
flow-matching pipeline.
"""

import json
import os
import random
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset

from src.datasets.transforms import get_train_transforms, get_val_transforms


def infer_sample_type_sota(scenario_name, view_type=None):
    """Classify sample type for SOTA pipeline."""
    if view_type == "front" or scenario_name.startswith("bdd_"):
        return "bdd_front"
    if view_type == "vehicle" or "vehicle_view" in scenario_name:
        return "wts_vehicle"
    return "wts_overhead"


def select_frame_paths_sota(paths, max_frames, strategy="dense_tail"):
    """Select bounded history window with SOTA strategies."""
    if len(paths) <= max_frames:
        return paths

    if strategy == "tail":
        return paths[-max_frames:]
    elif strategy == "uniform":
        step = len(paths) / max_frames
        indices = [int(i * step) for i in range(max_frames)]
        return [paths[i] for i in indices]
    elif strategy == "dense_tail":
        tail_count = int(max_frames * 0.75)
        head_count = max_frames - tail_count
        tail = paths[-tail_count:]
        head_pool = paths[:-tail_count]
        if head_count > 0 and len(head_pool) > 0:
            head_step = max(1, len(head_pool) // head_count)
            head = head_pool[::head_step][:head_count]
        else:
            head = []
        return head + tail
    else:
        return paths[-max_frames:]


class SOTAWTSDataset(Dataset):
    """
    SOTA dataset with dual caption support.

    Returns dict with:
        - history_frames: [T, C, H, W] tensor
        - future_frames: [T_future, C, H, W] tensor (if available)
        - caption_pedestrian: str
        - caption_vehicle: str
        - merged_caption: str
        - phase_label: int
        - scenario_name: str
        - sample_type: str
        - history_paths: list[str]
        - original_history_length: int
        - frame_length: int (for test samples)
    """

    def __init__(
        self,
        manifest_path,
        split="train",
        max_history_frames=16,
        max_future_frames=8,
        image_size=(288, 512),
        history_strategy="dense_tail",
        augment=False,
    ):
        self.samples = []
        self.split = split
        self.max_history_frames = max_history_frames
        self.max_future_frames = max_future_frames
        self.history_strategy = history_strategy

        with open(manifest_path, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    self.samples.append(json.loads(line))

        if split == "train" and augment:
            self.transforms = get_train_transforms(image_size)
        else:
            self.transforms = get_val_transforms(image_size)

        self.augment = augment

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        history_paths = sample["history_frames"]
        history_paths = select_frame_paths_sota(
            history_paths, self.max_history_frames, self.history_strategy
        )

        history_frames = []
        for path in history_paths:
            img = Image.open(path).convert("RGB")
            if self.transforms:
                img = self.transforms(img)
            history_frames.append(img)

        history_tensor = torch.stack(history_frames, dim=0)

        future_tensor = None
        if "future_frames" in sample and sample["future_frames"]:
            future_paths = sample["future_frames"][:self.max_future_frames]
            future_frames = []
            for path in future_paths:
                img = Image.open(path).convert("RGB")
                if self.transforms:
                    img = self.transforms(img)
                future_frames.append(img)
            future_tensor = torch.stack(future_frames, dim=0)

        if self.augment and random.random() > 0.5:
            history_tensor = torch.flip(history_tensor, dims=[-1])
            if future_tensor is not None:
                future_tensor = torch.flip(future_tensor, dims=[-1])

        caption_pedestrian = sample.get("caption_pedestrian", "")
        caption_vehicle = sample.get("caption_vehicle", "")
        merged_caption = sample.get("merged_caption", f"{caption_vehicle} {caption_pedestrian}")

        phase_label = sample.get("phase_label", 0)
        scenario_name = sample.get("scenario_name", "")
        view_type = sample.get("view_type", "overhead")
        sample_type = infer_sample_type_sota(scenario_name, view_type)
        frame_length = sample.get("frame_length", self.max_future_frames)

        result = {
            "history_frames": history_tensor,
            "caption_pedestrian": caption_pedestrian,
            "caption_vehicle": caption_vehicle,
            "merged_caption": merged_caption,
            "phase_label": torch.tensor(phase_label, dtype=torch.long),
            "scenario_name": scenario_name,
            "sample_type": sample_type,
            "history_paths": history_paths,
            "original_history_length": len(history_paths),
            "frame_length": frame_length,
        }

        if future_tensor is not None:
            result["future_frames"] = future_tensor

        return result

"""
src/data/wts_dataset.py
=======================
WTS (Watch, Track, and Segment) Dataset Loader for Track 5.

Dataset Structure
-----------------
The WTS dataset contains:
  - 810 videos across 155 scenarios
  - ~5 annotated segments per scenario
  - Each segment: event_phase labels + caption_pedestrian + caption_vehicle
  - Segment timing: start_time, end_time (float, seconds)

Annotation JSON Format:
    {
      "id": 722,
      "overhead_videos": [
        "20230707_8_SN46_T1_Camera1_0.mp4",
        ...
      ],
      "event_phase": [
        {
          "labels": ["4"],                    # event phase ID
          "caption_pedestrian": "...",
          "caption_vehicle": "...",
          "start_time": "39.395",
          "end_time": "44.663"
        }
      ]
    }

Event Phase Labels (from WTS):
    1: Pre-Recognition
    2: Recognition
    3: Judgment
    4: Action
    5: Avoidance

The task per segment:
  - Input:  last K frames of previous segment (or first K frames of current)
  - Target: remaining frames of current segment
  - Conditioning: caption_pedestrian + caption_vehicle

References
----------
- WTS Dataset: https://woven-planet.github.io/wts_dataset/
- Challenge: CVPR 2026 Track 5 - Generative Traffic Video Forecasting
"""

import os
import json
import math
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# WTS event phase names (for logging/debugging)
PHASE_NAMES = {
    "1": "Pre-Recognition",
    "2": "Recognition",
    "3": "Judgment",
    "4": "Action",
    "5": "Avoidance",
}

# Default number of history frames to extract as conditioning input
DEFAULT_NUM_HISTORY_FRAMES = 8

# Default number of future frames to predict
DEFAULT_NUM_TARGET_FRAMES = 33


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def load_video_frames(
    video_path: str,
    start_time: float,
    end_time: float,
    fps: Optional[float] = None,
) -> List[np.ndarray]:
    """
    Load video frames between start_time and end_time.
    
    Returns list of BGR numpy arrays (H, W, 3), uint8.
    
    Args:
        video_path: path to .mp4 file
        start_time: start time in seconds
        end_time: end time in seconds
        fps: if None, use video's native FPS
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_path}")
    
    native_fps = cap.get(cv2.CAP_PROP_FPS)
    if fps is None:
        fps = native_fps
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / native_fps
    
    start_frame = int(start_time * native_fps)
    end_frame = int(min(end_time * native_fps, total_frames - 1))
    
    # Seek to start frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    
    frames = []
    frame_idx = start_frame
    while frame_idx <= end_frame:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
        frame_idx += 1
    
    cap.release()
    return frames


def frames_to_tensor(
    frames: List[np.ndarray],
    resize: Optional[Tuple[int, int]] = None,
    normalize: bool = True,
) -> torch.Tensor:
    """
    Convert list of BGR numpy frames to a tensor of shape (T, C, H, W).
    
    Args:
        frames: list of (H, W, 3) BGR uint8 arrays
        resize: optional (H, W) for resizing
        normalize: if True, scale to [-1, 1] (Wan2.2 convention)
    
    Returns:
        Tensor of shape (T, 3, H, W), dtype float32
    """
    processed = []
    for frame in frames:
        # BGR → RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(frame_rgb)
        
        # Enforce Modulo 16 for Wan 2.2 VAE Compatibility
        if resize is not None:
            target_h, target_w = resize
        else:
            target_h, target_w = pil_img.size[1], pil_img.size[0]
            
        target_h = (target_h // 16) * 16
        target_w = (target_w // 16) * 16
        
        pil_img = pil_img.resize((target_w, target_h), Image.LANCZOS)
        
        arr = np.array(pil_img, dtype=np.float32) / 255.0  # [0, 1]
        if normalize:
            arr = arr * 2.0 - 1.0  # [-1, 1]
        
        tensor = torch.from_numpy(arr).permute(2, 0, 1)  # (3, H, W)
        processed.append(tensor)
    
    return torch.stack(processed, dim=0)  # (T, 3, H, W)


def sample_frames_uniform(
    frames: List[Any],
    n: int,
    from_end: bool = False,
) -> List[Any]:
    """
    Uniformly sample n frames from a list.
    
    Args:
        frames: source frames
        n: target count
        from_end: if True, sample from the end of the list
    
    Returns:
        List of n frames
    """
    if len(frames) == 0:
        return []
    if len(frames) <= n:
        return frames
    
    if from_end:
        # Take the last n frames (most recent history)
        return frames[-n:]
    
    indices = np.linspace(0, len(frames) - 1, n, dtype=int)
    return [frames[i] for i in indices]


# ---------------------------------------------------------------------------
# Main Dataset Class
# ---------------------------------------------------------------------------

class WTSDataset(Dataset):
    """
    PyTorch Dataset for WTS Track 5 training.
    
    Each sample represents one event phase segment:
      - history_frames: last K frames before current segment starts
      - target_frames: frames of current segment (ground truth)
      - caption_pedestrian: text describing pedestrian behavior
      - caption_vehicle: text describing vehicle behavior
      - metadata: segment timing, phase label, video path, etc.
    
    Args:
        annotation_json: path to WTS annotation JSON file
        video_root: root directory containing video files
        split: "train" | "val" | "test"
        num_history_frames: number of history frames (K)
        num_target_frames: number of target frames to predict (N)
        target_resolution: (H, W) to resize frames; None keeps original
        normalize: normalize frames to [-1, 1]
        augment: apply data augmentation
        
    Example:
        >>> dataset = WTSDataset(
        ...     annotation_json="data/wts/train_annotations.json",
        ...     video_root="data/wts/videos",
        ...     split="train",
        ...     num_history_frames=8,
        ...     num_target_frames=33,
        ... )
        >>> sample = dataset[0]
        >>> print(sample["history_frames"].shape)  # (8, 3, H, W)
        >>> print(sample["target_frames"].shape)   # (33, 3, H, W)
    """
    
    def __init__(
        self,
        annotation_json: str,
        video_root: str,
        split: str = "train",
        num_history_frames: int = DEFAULT_NUM_HISTORY_FRAMES,
        num_target_frames: int = DEFAULT_NUM_TARGET_FRAMES,
        target_resolution: Optional[Tuple[int, int]] = None,  # (H, W)
        normalize: bool = True,
        augment: bool = False,
        prev_segment_overlap: float = 0.2,  # fraction of prev segment to use as history
    ):
        self.video_root = Path(video_root)
        self.split = split
        self.num_history_frames = num_history_frames
        self.num_target_frames = num_target_frames
        self.target_resolution = target_resolution
        self.normalize = normalize
        self.augment = augment
        self.prev_segment_overlap = prev_segment_overlap
        
        # Load annotations
        with open(annotation_json, "r") as f:
            raw_annotations = json.load(f)
        
        # Build flat list of (scenario, segment_idx) pairs
        self.samples = self._build_sample_list(raw_annotations)
        
        print(f"[WTSDataset] Loaded {len(self.samples)} samples from {split} split")
    
    def _build_sample_list(
        self, annotations: List[Dict]
    ) -> List[Dict]:
        """
        Flatten annotation structure into a list of individual segment samples.
        
        For each segment, we record:
          - The video path(s) covering this segment
          - The time window of the *previous* segment (for history frames)
          - The time window of the *current* segment (for target frames)
        """
        samples = []
        
        for scenario in annotations:
            scenario_id = scenario["id"]
            video_paths = [
                str(self.video_root / v)
                for v in scenario.get("overhead_videos", [])
            ]
            phases = scenario.get("event_phase", [])
            
            for seg_idx, phase in enumerate(phases):
                # Skip segments without timing or captions
                if "start_time" not in phase or "end_time" not in phase:
                    continue
                if not phase.get("caption_pedestrian") or not phase.get("caption_vehicle"):
                    continue
                
                seg_start = float(phase["start_time"])
                seg_end = float(phase["end_time"])
                
                # History: last portion of previous segment (or beginning of current)
                if seg_idx > 0:
                    prev_phase = phases[seg_idx - 1]
                    prev_start = float(prev_phase["start_time"])
                    prev_end = float(prev_phase["end_time"])
                    prev_duration = prev_end - prev_start
                    # Use last 'prev_segment_overlap' fraction of previous segment
                    history_start = prev_end - prev_duration * self.prev_segment_overlap
                    history_end = prev_end
                else:
                    # First segment: use first K frames of current segment as history
                    history_start = seg_start
                    history_end = seg_start + 0.5  # rough estimate; adjusted at load time
                
                sample = {
                    "scenario_id": scenario_id,
                    "segment_idx": seg_idx,
                    "video_paths": video_paths,
                    "history_start": history_start,
                    "history_end": history_end,
                    "target_start": seg_start,
                    "target_end": seg_end,
                    "caption_pedestrian": phase["caption_pedestrian"],
                    "caption_vehicle": phase["caption_vehicle"],
                    "phase_label": phase.get("labels", ["?"])[0],
                }
                samples.append(sample)
        
        return samples
    
    def _find_video_for_time(
        self, video_paths: List[str], timestamp: float
    ) -> Optional[str]:
        """
        Find the video file that contains a given timestamp.
        
        In WTS, multiple camera clips are listed per scenario with timestamps
        encoded in the filename (e.g., Camera1_0.mp4 → camera 1, segment 0).
        We check each video's duration to find the right file.
        """
        # Try each video path; return first one that exists and is accessible
        for vp in video_paths:
            if os.path.isfile(vp):
                cap = cv2.VideoCapture(vp)
                if cap.isOpened():
                    fps = cap.get(cv2.CAP_PROP_FPS)
                    total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
                    duration = total_frames / fps if fps > 0 else 0
                    cap.release()
                    if duration > 0:
                        return vp  # Use first available; adjust if needed
        return video_paths[0] if video_paths else None
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Load one training sample.
        
        Returns
        -------
        dict with keys:
            history_frames  : (K, 3, H, W) tensor, normalized
            target_frames   : (N, 3, H, W) tensor, normalized
            last_history_frame : (3, H, W) tensor — used as I2V anchor
            caption_pedestrian : str
            caption_vehicle    : str
            metadata           : dict with scenario_id, phase, timing
        """
        sample = self.samples[idx]
        
        # Find the video file
        video_path = self._find_video_for_time(
            sample["video_paths"], sample["target_start"]
        )
        if video_path is None:
            raise FileNotFoundError(
                f"No video found for scenario {sample['scenario_id']}"
            )
        
        # Load history frames
        try:
            hist_frames_raw = load_video_frames(
                video_path,
                sample["history_start"],
                sample["history_end"],
            )
        except Exception as e:
            # Fallback: use first frames of target segment as history
            print(f"[WARNING] History load failed for scenario {sample['scenario_id']}: {e}")
            hist_frames_raw = []
        
        # Load target frames
        try:
            tgt_frames_raw = load_video_frames(
                video_path,
                sample["target_start"],
                sample["target_end"],
            )
        except Exception as e:
            raise RuntimeError(
                f"Target load failed for scenario {sample['scenario_id']}: {e}"
            )
        
        # Fallback for history: use first frames of target
        if len(hist_frames_raw) < 2:
            hist_frames_raw = tgt_frames_raw[:min(8, len(tgt_frames_raw) // 4)]
        
        # Sample fixed number of frames
        hist_frames = sample_frames_uniform(
            hist_frames_raw, self.num_history_frames, from_end=True
        )
        tgt_frames = sample_frames_uniform(
            tgt_frames_raw, self.num_target_frames, from_end=False
        )
        
        # Data augmentation (training only)
        if self.augment and self.split == "train":
            hist_frames, tgt_frames = self._augment(hist_frames, tgt_frames)
        
        # Convert to tensors
        history_tensor = frames_to_tensor(
            hist_frames, self.target_resolution, self.normalize
        )
        target_tensor = frames_to_tensor(
            tgt_frames, self.target_resolution, self.normalize
        )
        
        # Last history frame is the I2V anchor image
        last_history_frame = history_tensor[-1]  # (3, H, W)
        
        return {
            "history_frames": history_tensor,         # (K, 3, H, W)
            "target_frames": target_tensor,           # (N, 3, H, W)
            "last_history_frame": last_history_frame, # (3, H, W)
            "caption_pedestrian": sample["caption_pedestrian"],
            "caption_vehicle": sample["caption_vehicle"],
            "metadata": {
                "scenario_id": sample["scenario_id"],
                "segment_idx": sample["segment_idx"],
                "phase_label": sample["phase_label"],
                "phase_name": PHASE_NAMES.get(sample["phase_label"], "Unknown"),
                "target_start": sample["target_start"],
                "target_end": sample["target_end"],
                "video_path": video_path,
            },
        }
    
    def _augment(
        self,
        hist_frames: List[np.ndarray],
        tgt_frames: List[np.ndarray],
    ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        Apply consistent augmentation to history + target frames.
        
        Augmentations applied:
        - Color jitter (brightness/contrast/saturation) — same transform for all frames
        
        NOT applied:
        - Horizontal flip (traffic directionality matters; flipping breaks semantics)
        - Vertical flip (overhead view has fixed orientation)
        - Random crop (resolution must match for submission)
        """
        # Random brightness/contrast jitter applied consistently across all frames
        alpha = random.uniform(0.9, 1.1)   # contrast
        beta = random.randint(-10, 10)     # brightness (pixel offset)
        
        def apply_jitter(frame: np.ndarray) -> np.ndarray:
            frame = frame.astype(np.float32)
            frame = frame * alpha + beta
            return np.clip(frame, 0, 255).astype(np.uint8)
        
        hist_frames = [apply_jitter(f) for f in hist_frames]
        tgt_frames = [apply_jitter(f) for f in tgt_frames]
        
        return hist_frames, tgt_frames


# ---------------------------------------------------------------------------
# Inference-Time Test Dataset (no ground truth frames)
# ---------------------------------------------------------------------------

class WTSTestDataset(Dataset):
    """
    Dataset for inference on WTS test set.
    
    Each sample provides:
      - The history frame sequence (pre-provided PNG files)
      - Both captions
      - Output directory for generated frames
    
    Args:
        test_json: path to test annotation JSON
        history_dir: root directory containing history frame folders
        output_dir: where to write generated frames
    """
    
    def __init__(
        self,
        test_json: str,
        history_dir: str,
        output_dir: str,
        target_resolution: Optional[Tuple[int, int]] = None,
        normalize: bool = True,
    ):
        self.history_dir = Path(history_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.target_resolution = target_resolution
        self.normalize = normalize
        
        with open(test_json, "r") as f:
            self.test_cases = json.load(f)
        
        print(f"[WTSTestDataset] Loaded {len(self.test_cases)} test cases")
    
    def __len__(self) -> int:
        return len(self.test_cases)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        case = self.test_cases[idx]
        case_id = case["id"]
        
        # Load history frames from directory (pre-extracted PNGs)
        hist_dir = self.history_dir / str(case_id)
        hist_frame_paths = sorted(hist_dir.glob("*.png")) + sorted(hist_dir.glob("*.jpg"))
        
        if len(hist_frame_paths) == 0:
            raise FileNotFoundError(
                f"No history frames found in {hist_dir}"
            )
        
        hist_frames = []
        for p in hist_frame_paths:
            img = cv2.imread(str(p))
            if img is not None:
                hist_frames.append(img)
        
        # Get original resolution from first frame (submission must match this)
        h, w = hist_frames[0].shape[:2]
        original_resolution = (h, w)
        
        # Convert to tensor
        history_tensor = frames_to_tensor(
            hist_frames, self.target_resolution, self.normalize
        )
        
        last_history_frame = history_tensor[-1]
        
        # Output directory for this test case
        case_output_dir = self.output_dir / str(case_id)
        case_output_dir.mkdir(parents=True, exist_ok=True)
        
        return {
            "case_id": case_id,
            "history_frames": history_tensor,
            "last_history_frame": last_history_frame,
            "caption_pedestrian": case.get("caption_pedestrian", ""),
            "caption_vehicle": case.get("caption_vehicle", ""),
            "original_resolution": original_resolution,
            "output_dir": str(case_output_dir),
            "num_target_frames": case.get("num_target_frames", DEFAULT_NUM_TARGET_FRAMES),
        }

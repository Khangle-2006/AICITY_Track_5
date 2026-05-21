import torch


def wts_collate_fn(batch):
    """Custom collate function for WTS samples."""
    history_frames = torch.stack([item["history_frames"] for item in batch], dim=0)
    phase_labels = torch.stack([item["phase_label"] for item in batch], dim=0)
    merged_captions = [item.get("merged_caption", "") for item in batch]

    res = {
        "history_frames": history_frames,
        "merged_caption": merged_captions,
        "phase_label": phase_labels,
        "scenario_name": [item.get("scenario_name", "unknown") for item in batch],
        "sample_type": [item.get("sample_type", "unknown") for item in batch],
    }

    if "caption_pedestrian" in batch[0]:
        res["caption_pedestrian"] = [item["caption_pedestrian"] for item in batch]

    if "caption_vehicle" in batch[0]:
        res["caption_vehicle"] = [item["caption_vehicle"] for item in batch]

    if "history_paths" in batch[0]:
        res["history_paths"] = [item["history_paths"] for item in batch]

    if "original_history_length" in batch[0]:
        res["original_history_length"] = [item["original_history_length"] for item in batch]

    if "frame_length" in batch[0]:
        res["frame_length"] = [item["frame_length"] for item in batch]

    if "future_frames" in batch[0]:
        future_frames = torch.stack([item["future_frames"] for item in batch], dim=0)
        res["future_frames"] = future_frames

    return res

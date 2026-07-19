from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class TicftLayout:
    condition_frames: int
    buffer_frames: int
    target_frames: int
    loss_on: str = "target"

    @property
    def total_frames(self) -> int:
        return self.condition_frames + self.buffer_frames + self.target_frames

    @property
    def condition_slice(self) -> tuple[int, int]:
        return 0, self.condition_frames

    @property
    def buffer_slice(self) -> tuple[int, int]:
        start = self.condition_frames
        return start, start + self.buffer_frames

    @property
    def target_slice(self) -> tuple[int, int]:
        start = self.condition_frames + self.buffer_frames
        return start, start + self.target_frames


def layout_from_manifest_row(row: dict[str, Any]) -> TicftLayout:
    cfg = row.get("ticft", {})
    return TicftLayout(
        condition_frames=int(cfg.get("condition_frames", 0) or 0),
        buffer_frames=int(cfg.get("buffer_frames", 0) or 0),
        target_frames=int(cfg.get("target_frames", 0) or 0),
        loss_on=str(cfg.get("loss_on", "target")),
    )


def buffer_noise_levels(buffer_frames: int, max_timestep: int = 1000) -> list[int]:
    if buffer_frames <= 0:
        return []
    return [round((i + 1) * max_timestep / (buffer_frames + 1)) for i in range(buffer_frames)]


def describe_layout(row: dict[str, Any], max_timestep: int = 1000) -> dict[str, Any]:
    layout = layout_from_manifest_row(row)
    return {
        "condition_slice": layout.condition_slice,
        "buffer_slice": layout.buffer_slice,
        "target_slice": layout.target_slice,
        "total_frames": layout.total_frames,
        "buffer_noise_levels": buffer_noise_levels(layout.buffer_frames, max_timestep),
        "loss_on": layout.loss_on,
    }


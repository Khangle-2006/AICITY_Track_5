from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .layout import TicftLayout, layout_from_manifest_row


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def frame_paths(row: dict[str, Any]) -> tuple[list[Path], list[Path]]:
    frames = row.get("frames", {})
    condition = [Path(p) for p in frames.get("condition", [])]
    target = [Path(p) for p in frames.get("target", [])]
    return condition, target


def validate_row(row: dict[str, Any], require_target: bool = True) -> list[str]:
    errors: list[str] = []
    layout = layout_from_manifest_row(row)
    condition, target = frame_paths(row)

    if row.get("status") != "ok":
        errors.append(f"status={row.get('status')}")
    if len(condition) != layout.condition_frames:
        errors.append(f"condition_count={len(condition)}/{layout.condition_frames}")
    if require_target and len(target) != layout.target_frames:
        errors.append(f"target_count={len(target)}/{layout.target_frames}")

    for path in condition + (target if require_target else []):
        if not path.exists():
            errors.append(f"missing={path}")
        elif path.stat().st_size == 0:
            errors.append(f"zero_byte={path}")
    if not str(row.get("prompt", "")).strip():
        errors.append("empty_prompt")
    return errors


def summarize_manifest(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_view: dict[str, int] = {}
    by_dataset: dict[str, int] = {}
    by_phase: dict[str, int] = {}
    prompt_words = []
    layouts: dict[str, int] = {}

    for row in rows:
        by_view[row.get("view_type", "")] = by_view.get(row.get("view_type", ""), 0) + 1
        by_dataset[row.get("dataset_part", "")] = by_dataset.get(row.get("dataset_part", ""), 0) + 1
        by_phase[str(row.get("phase", ""))] = by_phase.get(str(row.get("phase", "")), 0) + 1
        prompt_words.append(len(str(row.get("prompt", "")).split()))
        layout = layout_from_manifest_row(row)
        key = f"c{layout.condition_frames}_b{layout.buffer_frames}_t{layout.target_frames}"
        layouts[key] = layouts.get(key, 0) + 1

    return {
        "rows": len(rows),
        "by_view": by_view,
        "by_dataset": by_dataset,
        "by_phase": by_phase,
        "layouts": layouts,
        "prompt_words_mean": sum(prompt_words) / max(len(prompt_words), 1),
        "prompt_words_min": min(prompt_words) if prompt_words else 0,
        "prompt_words_max": max(prompt_words) if prompt_words else 0,
    }


class ManifestDataset:
    """Thin manifest dataset for Wan backend integration."""

    def __init__(self, jsonl: Path, require_target: bool = True):
        self.jsonl = jsonl
        self.rows = read_jsonl(jsonl)
        self.require_target = require_target

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        condition, target = frame_paths(row)
        layout: TicftLayout = layout_from_manifest_row(row)
        return {
            "row": row,
            "sample_name": row.get("sample_name") or row.get("sample_id"),
            "prompt": row.get("prompt", ""),
            "condition_paths": condition,
            "target_paths": target,
            "layout": layout,
        }

    def validation_errors(self, max_bad: int = 20) -> list[tuple[str, list[str]]]:
        bad = []
        for row in self.rows:
            errors = validate_row(row, self.require_target)
            if errors:
                bad.append((str(row.get("sample_name") or row.get("sample_id")), errors))
            if len(bad) >= max_bad:
                break
        return bad


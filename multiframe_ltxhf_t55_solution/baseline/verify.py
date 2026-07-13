from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .frame_io import numeric_pngs


@dataclass
class VerifyResult:
    ok: bool
    frames_found: int
    missing_count: int
    notes: str


def verify_sample_output(output_dir: Path, target_n: int) -> VerifyResult:
    pngs = numeric_pngs(output_dir)
    found = {int(p.stem) for p in pngs}
    expected = set(range(target_n))
    missing = sorted(expected - found)
    extra = sorted(found - expected)
    notes = []
    if missing:
        notes.append(f"missing={missing[:10]}")
    if extra:
        notes.append(f"extra={extra[:10]}")
    return VerifyResult(ok=(not missing and not extra), frames_found=len(pngs), missing_count=len(missing), notes="; ".join(notes))

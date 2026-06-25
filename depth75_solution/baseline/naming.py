from __future__ import annotations


def parse_int(value: str, default: int = 0) -> int:
    try:
        if value == "":
            return default
        return int(float(value))
    except Exception:
        return default


def safe_name(value: str) -> str:
    keep = []
    for ch in value:
        if ch.isalnum() or ch in ("-", "_", "."):
            keep.append(ch)
        else:
            keep.append("_")
    out = "".join(keep).strip("._")
    return out or "sample"


def resolve_output_name(row: dict[str, str], used: set[str]) -> str:
    base = safe_name(row.get("source_folder") or row.get("sample_id") or "sample")
    if base not in used:
        used.add(base)
        return base

    fallback = safe_name(row.get("sample_id") or base)
    if fallback not in used:
        used.add(fallback)
        return fallback

    i = 2
    while f"{fallback}_{i}" in used:
        i += 1
    name = f"{fallback}_{i}"
    used.add(name)
    return name

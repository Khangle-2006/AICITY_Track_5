from __future__ import annotations

import argparse
import hashlib
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify a submission against a SHA-256 manifest.")
    parser.add_argument("--submission", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--write", action="store_true", help="Write a manifest instead of verifying one.")
    return parser.parse_args()


def digest(path: Path) -> str:
    checksum = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            checksum.update(block)
    return checksum.hexdigest()


def files_below(root: Path) -> list[Path]:
    return sorted(path for path in root.rglob("*") if path.is_file())


def write_manifest(root: Path, manifest: Path) -> None:
    files = files_below(root)
    manifest.parent.mkdir(parents=True, exist_ok=True)
    with manifest.open("w", encoding="utf-8", newline="\n") as handle:
        for position, path in enumerate(files, start=1):
            relative = path.relative_to(root).as_posix()
            handle.write(f"{digest(path)}  {relative}\n")
            if position % 100 == 0 or position == len(files):
                print(f"hashed={position}/{len(files)}", flush=True)


def read_manifest(path: Path) -> dict[str, str]:
    expected: dict[str, str] = {}
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            checksum, relative = line.split("  ", 1)
        except ValueError as exc:
            raise RuntimeError(f"invalid manifest line {line_number}") from exc
        expected[relative] = checksum
    return expected


def verify(root: Path, manifest: Path) -> None:
    expected = read_manifest(manifest)
    actual_paths = {path.relative_to(root).as_posix(): path for path in files_below(root)}
    missing = sorted(set(expected) - set(actual_paths))
    unexpected = sorted(set(actual_paths) - set(expected))
    mismatched: list[str] = []

    for position, (relative, checksum) in enumerate(expected.items(), start=1):
        path = actual_paths.get(relative)
        if path is not None and digest(path) != checksum:
            mismatched.append(relative)
        if position % 100 == 0 or position == len(expected):
            print(f"verified={position}/{len(expected)}", flush=True)

    if missing or unexpected or mismatched:
        for label, values in (("missing", missing), ("unexpected", unexpected), ("mismatched", mismatched)):
            if values:
                print(f"{label}={len(values)}")
                for value in values[:20]:
                    print(f"  {value}")
        raise SystemExit(2)

    print(f"status=exact-match files={len(expected)}")


def main() -> None:
    args = parse_args()
    if not args.submission.is_dir():
        raise FileNotFoundError(args.submission)
    if args.write:
        write_manifest(args.submission, args.manifest)
    else:
        verify(args.submission, args.manifest)


if __name__ == "__main__":
    main()

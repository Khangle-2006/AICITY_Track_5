from __future__ import annotations

from typing import Any


def print_line(char: str = "=", width: int = 100) -> None:
    print(char * width)


def section(title: str) -> None:
    print()
    print_line("=")
    print(title)
    print_line("=")


def kv(key: str, value: Any, indent: int = 0) -> None:
    print(f"{' ' * indent}{key:<42} {value}")

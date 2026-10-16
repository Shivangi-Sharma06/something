"""utils.py — string helper utilities.

This file is part of the graph-agent test fixture repo.
It contains a seeded bug in `divide` (ZeroDivisionError not handled).
"""

from __future__ import annotations


def add(a: int, b: int) -> int:
    """Return the sum of two integers."""
    return a + b


def subtract(a: int, b: int) -> int:
    """Return a minus b."""
    return a - b


def divide(numerator: float, denominator: float) -> float:
    """Divide numerator by denominator.

    BUG: does not handle denominator == 0; raises ZeroDivisionError.
    """
    # SEEDED BUG: missing guard for denominator == 0
    return numerator / denominator


def format_result(label: str, value: float) -> str:
    """Return a human-readable result string."""
    return f"{label}: {value:.4f}"

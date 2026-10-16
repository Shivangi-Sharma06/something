"""tests/test_utils.py — Tests for the sample fixture repo.

The test_divide_by_zero test FAILS on the unpatched code (seeded bug).
This is intentional — it is the "failing test" input for bug localization.
"""

from __future__ import annotations

import pytest
from utils import add, subtract, divide, format_result


def test_add():
    assert add(2, 3) == 5
    assert add(-1, 1) == 0


def test_subtract():
    assert subtract(10, 3) == 7


def test_divide_normal():
    assert divide(10.0, 2.0) == pytest.approx(5.0)


def test_divide_by_zero():
    """This test FAILS on the buggy code — used as the bug-localization input."""
    with pytest.raises(ZeroDivisionError):
        divide(10.0, 0.0)
    # After the patch, divide(10.0, 0.0) should return float('inf') or raise ValueError
    # (the exact fix is left to the LLM to propose)


def test_format_result():
    assert format_result("x", 3.14159) == "x: 3.1416"

"""main.py — entry point for the sample fixture repo.

Calls utilities from utils.py; deliberately calls divide() with zero denominator
to trigger the seeded bug.
"""

from __future__ import annotations

from utils import add, divide, format_result


class Calculator:
    """Simple calculator class that delegates to utility functions."""

    def __init__(self, precision: int = 4):
        self.precision = precision

    def run_add(self, a: int, b: int) -> str:
        result = add(a, b)
        return format_result("sum", result)

    def run_divide(self, a: float, b: float) -> str:
        """Divide a by b and format the result."""
        result = divide(a, b)
        return format_result("quotient", result)


def main() -> None:
    calc = Calculator()
    print(calc.run_add(10, 5))
    # BUG TRIGGER: calling divide with b=0 will raise ZeroDivisionError
    print(calc.run_divide(10.0, 0.0))


if __name__ == "__main__":
    main()

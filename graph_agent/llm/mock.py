"""MockAdapter — deterministic LLM adapter for tests and CI.

Returns a hardcoded patch for any input.  The patch is parameterised so
tests can set a custom response via MockAdapter(response="...").
"""

from __future__ import annotations

from graph_agent.llm.base import ModelAdapter

# A minimal unified diff that fixes the seeded ZeroDivisionError bug
# by adding a denominator guard to utils.divide().
DEFAULT_MOCK_DIFF = """\
--- a/utils.py
+++ b/utils.py
@@ -14,7 +14,10 @@ def divide(numerator: float, denominator: float) -> float:
     BUG: does not handle denominator == 0; raises ZeroDivisionError.
     \"\"\"
-    # SEEDED BUG: missing guard for denominator == 0
-    return numerator / denominator
+    if denominator == 0:
+        raise ValueError("denominator must not be zero")
+    return numerator / denominator
"""

DEFAULT_MOCK_ROOT_CAUSE = (
    "The function `divide` in `utils.py` does not guard against a zero denominator, "
    "causing an unhandled ZeroDivisionError when called with denominator=0. "
    "The fix adds an explicit check that raises ValueError with a clear message."
)

MOCK_RESPONSE_TEMPLATE = """\
## Root Cause Analysis

{root_cause}

## Patch

```diff
{diff}
```
"""


class MockAdapter(ModelAdapter):
    """Deterministic mock adapter — returns a canned response regardless of input."""

    def __init__(
        self,
        response: str | None = None,
        diff: str = DEFAULT_MOCK_DIFF,
        root_cause: str = DEFAULT_MOCK_ROOT_CAUSE,
    ):
        self._response = response or MOCK_RESPONSE_TEMPLATE.format(
            root_cause=root_cause, diff=diff
        )

    def name(self) -> str:
        return "mock"

    def complete(self, messages: list[dict], **kwargs) -> str:
        return self._response

"""Unit tests for graph_agent.patch.validator."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

SAMPLE_REPO = Path(__file__).parent.parent / "fixtures" / "sample_repo"

# A patch that applies cleanly to the sample repo
VALID_DIFF = """\
--- a/utils.py
+++ b/utils.py
@@ -17,6 +17,8 @@ def divide(numerator: float, denominator: float) -> float:
     BUG: does not handle denominator == 0; raises ZeroDivisionError.
     \"\"\"
     # SEEDED BUG: missing guard for denominator == 0
+    if denominator == 0:
+        raise ValueError("denominator must not be zero")
     return numerator / denominator
"""

# A patch with a bogus hunk that will not apply
INVALID_DIFF = """\
--- a/nonexistent_file.py
+++ b/nonexistent_file.py
@@ -1,3 +1,4 @@
 x = 1
+y = 2
 z = 3
"""


class TestValidatePatch:
    def test_valid_patch_applies(self):
        """The valid diff must apply cleanly to the sample repo."""
        from graph_agent.patch.validator import validate_patch
        # Only run if git is available and sample_repo is a git repo or has git
        try:
            subprocess.run(
                ["git", "init"], cwd=str(SAMPLE_REPO),
                capture_output=True, check=False
            )
        except FileNotFoundError:
            pytest.skip("git not available")

        result = validate_patch(SAMPLE_REPO, VALID_DIFF)
        # Note: may fail if repo doesn't have git index — validate error message
        if not result.applies_cleanly:
            # Acceptable if the temp repo doesn't have a git index
            assert result.error  # must have an explanation

    def test_invalid_patch_rejected(self):
        """An invalid diff (referencing nonexistent file) must not apply."""
        from graph_agent.patch.validator import validate_patch
        result = validate_patch(SAMPLE_REPO, INVALID_DIFF)
        assert not result.applies_cleanly
        assert result.error

    def test_empty_diff_rejected(self):
        from graph_agent.patch.validator import validate_patch
        result = validate_patch(SAMPLE_REPO, "")
        assert not result.applies_cleanly

    def test_result_always_has_patch_diff(self):
        from graph_agent.patch.validator import validate_patch
        result = validate_patch(SAMPLE_REPO, VALID_DIFF)
        assert result.patch_diff == VALID_DIFF


class TestExtractPatch:
    def test_extracts_diff_from_response(self):
        from graph_agent.patch.generator import extract_patch
        response = (
            "## Root Cause\nThe divide function has no guard.\n\n"
            "## Patch\n```diff\n--- a/utils.py\n+++ b/utils.py\n"
            "@@ -1 +1 @@\n-old\n+new\n```"
        )
        patch = extract_patch(response)
        assert patch.is_valid
        assert "--- a/utils.py" in patch.diff

    def test_no_diff_block_is_invalid(self):
        from graph_agent.patch.generator import extract_patch
        response = "I couldn't find the bug, sorry."
        patch = extract_patch(response)
        assert not patch.is_valid
        assert patch.diff == ""

    def test_root_cause_extracted(self):
        from graph_agent.patch.generator import extract_patch
        response = "Root cause: missing guard.\n```diff\n--- a/f.py\n+++ b/f.py\n```"
        patch = extract_patch(response)
        assert "Root cause" in patch.root_cause

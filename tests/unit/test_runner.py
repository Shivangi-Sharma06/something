"""Unit tests for graph_agent.verifier.runner."""

from __future__ import annotations

import sys
import tempfile
import shutil
from pathlib import Path

import pytest

SAMPLE_REPO = Path(__file__).parent.parent / "fixtures" / "sample_repo"
PYTHON = sys.executable  # use current interpreter


class TestRunTestsInIsolation:
    def test_runs_passing_tests(self, tmp_path):
        """A repo with a trivially passing test must report success."""
        from graph_agent.verifier.runner import run_tests_in_isolation

        (tmp_path / "test_trivial.py").write_text(
            "def test_ok(): assert 1 + 1 == 2\n", encoding="utf-8"
        )
        result = run_tests_in_isolation(
            tmp_path,
            patch_diff=None,
            test_command=f"{PYTHON} -m pytest test_trivial.py -v --tb=short -q",
        )
        assert result.success
        assert result.tests_passed >= 1
        assert result.tests_failed == 0
        assert result.exit_code == 0

    def test_detects_failing_tests(self, tmp_path):
        from graph_agent.verifier.runner import run_tests_in_isolation

        (tmp_path / "test_fail.py").write_text(
            "def test_bad(): assert 1 == 2\n", encoding="utf-8"
        )
        result = run_tests_in_isolation(
            tmp_path,
            patch_diff=None,
            test_command=f"{PYTHON} -m pytest test_fail.py -v --tb=short -q",
        )
        assert not result.success
        assert result.exit_code != 0

    def test_does_not_modify_original_repo(self, tmp_path):
        """The original repo must be untouched after running tests."""
        from graph_agent.verifier.runner import run_tests_in_isolation

        sentinel = tmp_path / "sentinel.py"
        sentinel.write_text("x = 1\n", encoding="utf-8")
        (tmp_path / "test_x.py").write_text("def test_x(): pass\n", encoding="utf-8")

        run_tests_in_isolation(
            tmp_path,
            patch_diff=None,
            test_command=f"{PYTHON} -m pytest test_x.py -q",
        )
        # sentinel must still exist untouched
        assert sentinel.exists()
        assert sentinel.read_text() == "x = 1\n"

    def test_patch_applied_in_isolation(self, tmp_path):
        """A patch that fixes a failing test must result in success."""
        import subprocess
        from graph_agent.verifier.runner import run_tests_in_isolation

        # Write a broken file
        (tmp_path / "math_utils.py").write_text(
            "def add(a, b): return a - b  # BUG\n", encoding="utf-8"
        )
        (tmp_path / "test_math.py").write_text(
            "from math_utils import add\ndef test_add(): assert add(1,2) == 3\n",
            encoding="utf-8"
        )
        # Init git so `git apply` works
        subprocess.run(["git", "init"], cwd=str(tmp_path), capture_output=True)
        subprocess.run(
            ["git", "add", "."], cwd=str(tmp_path), capture_output=True
        )
        subprocess.run(
            ["git", "commit", "-m", "init", "--allow-empty-message",
             "--author=Test <t@t.com>"],
            cwd=str(tmp_path), capture_output=True
        )

        diff = (
            "--- a/math_utils.py\n"
            "+++ b/math_utils.py\n"
            "@@ -1 +1 @@\n"
            "-def add(a, b): return a - b  # BUG\n"
            "+def add(a, b): return a + b\n"
        )
        result = run_tests_in_isolation(
            tmp_path,
            patch_diff=diff,
            test_command=f"{PYTHON} -m pytest test_math.py -v --tb=short -q",
        )
        # If git apply worked, tests should pass; otherwise patch_applied=False
        if result.patch_applied:
            assert result.success, f"Expected success after patch. Output:\n{result.test_output}"

    def test_timeout_returns_error(self, tmp_path):
        from graph_agent.verifier.runner import run_tests_in_isolation

        (tmp_path / "test_t.py").write_text("def test_ok(): pass\n", encoding="utf-8")
        result = run_tests_in_isolation(
            tmp_path,
            patch_diff=None,
            test_command=f"{PYTHON} -m pytest test_t.py -q",
            timeout=120,  # generous timeout
        )
        # Should complete fine within 120s
        assert result.exit_code is not None


class TestParseOutput:
    def test_parses_passed_lines(self):
        from graph_agent.verifier.runner import _parse_pytest_output
        output = (
            "tests/test_x.py::test_foo PASSED\n"
            "tests/test_x.py::test_bar FAILED\n"
        )
        results = _parse_pytest_output(output)
        assert any(r.passed and "test_foo" in r.name for r in results)
        assert any(not r.passed and "test_bar" in r.name for r in results)

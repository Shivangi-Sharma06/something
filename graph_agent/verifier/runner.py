"""Test runner — executes the test suite in an isolated environment and captures results.

Isolation strategy: git worktree + temp venv (default).
Docker path is behind config flag `verifier.isolation = "docker"`.

Retry loop:
  - Max retries from config (verifier.max_retries).
  - On each failure, the test output is fed back to the LLM for another attempt.
  - A patch is NEVER committed without explicit human approval.
"""

from __future__ import annotations

import logging
import shlex
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

from graph_agent.settings import get_settings

log = logging.getLogger(__name__)


@dataclass
class TestResult:
    name: str
    passed: bool
    output: str


@dataclass
class VerificationResult:
    success: bool
    patch_applied: bool
    tests_passed: int
    tests_failed: int
    test_output: str
    per_test: list[TestResult] = field(default_factory=list)
    exit_code: int = -1
    error: str = ""


def _parse_pytest_output(output: str) -> list[TestResult]:
    """Parse pytest output (both -v and -q formats) into per-test results."""
    import re
    results = []
    verbose_found = False

    for line in output.splitlines():
        line_stripped = line.strip()
        # Verbose format: "tests/foo.py::test_bar PASSED"
        if " PASSED" in line_stripped:
            name = line_stripped.split(" PASSED")[0].strip()
            results.append(TestResult(name=name, passed=True, output=""))
            verbose_found = True
        elif " FAILED" in line_stripped:
            name = line_stripped.split(" FAILED")[0].strip()
            results.append(TestResult(name=name, passed=False, output=""))
            verbose_found = True
        elif " ERROR" in line_stripped and "::" in line_stripped:
            name = line_stripped.split(" ERROR")[0].strip()
            results.append(TestResult(name=name, passed=False, output="COLLECTION ERROR"))
            verbose_found = True

    # Quiet format fallback: parse "N passed, M failed" summary line
    if not verbose_found:
        summary_match = re.search(
            r'(\d+) passed(?:,\s*(\d+) failed)?', output
        )
        if summary_match:
            passed_n = int(summary_match.group(1))
            failed_n = int(summary_match.group(2) or 0)
            for i in range(passed_n):
                results.append(TestResult(name=f"test_{i}", passed=True, output=""))
            for i in range(failed_n):
                results.append(TestResult(name=f"failed_{i}", passed=False, output=""))

    return results


def run_tests_in_isolation(
    repo_root: str | Path,
    patch_diff: str | None = None,
    test_command: str | None = None,
    timeout: int | None = None,
) -> VerificationResult:
    """Apply *patch_diff* (if given) to a temp copy of *repo_root* and run tests.

    Always uses a throw-away temp directory; original repo is never modified.
    """
    s = get_settings().verifier
    cmd = test_command or s.test_command
    timeout_s = timeout or s.test_timeout_seconds
    repo_root = Path(repo_root).resolve()

    tmp_dir = Path(tempfile.mkdtemp(prefix="graph_agent_verify_"))
    try:
        # Copy repo to isolated dir
        tmp_repo = tmp_dir / "repo"
        shutil.copytree(str(repo_root), str(tmp_repo), symlinks=True)

        patch_applied = False
        if patch_diff:
            patch_file = tmp_repo / "_fix.patch"
            patch_file.write_text(patch_diff, encoding="utf-8")
            apply_proc = subprocess.run(
                ["git", "apply", "--ignore-whitespace", "_fix.patch"],
                cwd=str(tmp_repo),
                capture_output=True,
                text=True,
                timeout=30,
            )
            patch_file.unlink(missing_ok=True)
            if apply_proc.returncode != 0:
                err = (apply_proc.stderr or apply_proc.stdout).strip()
                log.warning("Patch failed to apply in isolation: %s", err)
                return VerificationResult(
                    success=False,
                    patch_applied=False,
                    tests_passed=0,
                    tests_failed=0,
                    test_output=err,
                    exit_code=apply_proc.returncode,
                    error=f"Patch apply failed: {err}",
                )
            patch_applied = True

        # Run the test command
        log.info("Running tests in %s: %s", tmp_repo, cmd)
        proc = subprocess.run(
            shlex.split(cmd),
            cwd=str(tmp_repo),
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )

        output = proc.stdout + "\n" + proc.stderr
        per_test = _parse_pytest_output(output)
        passed = sum(1 for t in per_test if t.passed)
        failed = sum(1 for t in per_test if not t.passed)

        return VerificationResult(
            success=proc.returncode == 0,
            patch_applied=patch_applied,
            tests_passed=passed,
            tests_failed=failed,
            test_output=output.strip(),
            per_test=per_test,
            exit_code=proc.returncode,
        )

    except subprocess.TimeoutExpired:
        return VerificationResult(
            success=False,
            patch_applied=False,
            tests_passed=0,
            tests_failed=0,
            test_output="",
            exit_code=-1,
            error=f"Test run timed out after {timeout_s}s",
        )
    except Exception as exc:
        return VerificationResult(
            success=False,
            patch_applied=False,
            tests_passed=0,
            tests_failed=0,
            test_output="",
            exit_code=-1,
            error=str(exc),
        )
    finally:
        shutil.rmtree(str(tmp_dir), ignore_errors=True)


def repair_loop(
    repo_root: str | Path,
    generate_fn,          # callable(attempt: int, prev_output: str) → GeneratedPatch
    validate_fn,          # callable(diff: str) → ValidationResult
    max_retries: int | None = None,
    test_command: str | None = None,
) -> tuple[VerificationResult, list[str]]:
    """Run the generate→validate→test→retry loop.

    Args:
        repo_root:    Repository root path.
        generate_fn:  Callable(attempt_num, prev_test_output) → GeneratedPatch
        validate_fn:  Callable(diff) → ValidationResult
        max_retries:  Max attempts (overrides config).
        test_command: Test command (overrides config).

    Returns:
        (final_VerificationResult, list_of_diffs_tried)
    """
    s = get_settings().verifier
    retries = max_retries if max_retries is not None else s.max_retries
    diffs_tried: list[str] = []
    prev_output = ""

    for attempt in range(retries):
        log.info("Repair attempt %d / %d", attempt + 1, retries)
        patch = generate_fn(attempt, prev_output)

        if not patch.is_valid:
            log.warning("Attempt %d: LLM returned no valid diff", attempt + 1)
            prev_output = "LLM did not produce a valid unified diff. Please try again."
            continue

        val = validate_fn(patch.diff)
        if not val.applies_cleanly:
            log.warning("Attempt %d: patch does not apply: %s", attempt + 1, val.error)
            prev_output = f"Patch failed to apply: {val.error}\nPlease revise the diff."
            continue

        diffs_tried.append(patch.diff)
        result = run_tests_in_isolation(repo_root, patch_diff=patch.diff, test_command=test_command)
        prev_output = result.test_output

        if result.success:
            log.info("All tests passed on attempt %d!", attempt + 1)
            return result, diffs_tried

        log.warning(
            "Attempt %d: tests failed (%d failed). Retrying ...",
            attempt + 1, result.tests_failed,
        )

    log.error("All %d repair attempts exhausted without success.", retries)
    return VerificationResult(
        success=False,
        patch_applied=False,
        tests_passed=0,
        tests_failed=0,
        test_output=prev_output,
        error=f"All {retries} repair attempts exhausted",
    ), diffs_tried

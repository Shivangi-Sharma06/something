"""Patch validator — applies diffs in an isolated git worktree and runs git apply --check.

ADR-6: We NEVER auto-accept a patch.  The validator only checks if the patch
applies cleanly; actual test execution is handled by verifier/runner.py.
A patch that applies cleanly can still introduce bugs — test results are the
gate for acceptance.
"""

from __future__ import annotations

import logging
import subprocess
import tempfile
import shutil
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    patch_diff: str
    applies_cleanly: bool
    error: str


def validate_patch(repo_root: str | Path, diff: str) -> ValidationResult:
    """Check if *diff* applies cleanly to *repo_root* using `git apply --check`.

    Uses a temporary copy of the repo so the original is never modified.
    Returns ValidationResult with applies_cleanly=True only if the patch
    applies without errors.
    """
    repo_root = Path(repo_root).resolve()
    tmp_dir = Path(tempfile.mkdtemp(prefix="graph_agent_patch_"))
    tmp_patch = tmp_dir / "fix.patch"

    try:
        # Copy the repo to an isolated temp dir
        tmp_repo = tmp_dir / "repo"
        shutil.copytree(str(repo_root), str(tmp_repo), symlinks=True)

        # Write the patch file
        tmp_patch_in_repo = tmp_repo / "fix.patch"
        tmp_patch_in_repo.write_text(diff, encoding="utf-8")

        # Run git apply --check
        proc = subprocess.run(
            ["git", "apply", "--check", "--ignore-whitespace", "fix.patch"],
            cwd=str(tmp_repo),
            capture_output=True,
            text=True,
            timeout=30,
        )

        if proc.returncode == 0:
            log.info("Patch applies cleanly to %s", repo_root)
            return ValidationResult(patch_diff=diff, applies_cleanly=True, error="")
        else:
            err = (proc.stderr or proc.stdout).strip()
            log.warning("Patch does NOT apply: %s", err)
            return ValidationResult(patch_diff=diff, applies_cleanly=False, error=err)

    except subprocess.TimeoutExpired:
        return ValidationResult(
            patch_diff=diff, applies_cleanly=False,
            error="git apply --check timed out"
        )
    except Exception as exc:
        return ValidationResult(
            patch_diff=diff, applies_cleanly=False,
            error=f"Unexpected error: {exc}"
        )
    finally:
        shutil.rmtree(str(tmp_dir), ignore_errors=True)


def apply_patch(repo_root: str | Path, diff: str) -> tuple[bool, str]:
    """Actually apply *diff* to *repo_root*.

    IMPORTANT: Only call this after explicit human approval.
    Returns (success, error_message).
    """
    repo_root = Path(repo_root).resolve()
    patch_file = repo_root / "_graph_agent_fix.patch"
    try:
        patch_file.write_text(diff, encoding="utf-8")
        proc = subprocess.run(
            ["git", "apply", "--ignore-whitespace", str(patch_file)],
            cwd=str(repo_root),
            capture_output=True,
            text=True,
            timeout=30,
        )
        patch_file.unlink(missing_ok=True)
        if proc.returncode == 0:
            return True, ""
        return False, (proc.stderr or proc.stdout).strip()
    except Exception as exc:
        patch_file.unlink(missing_ok=True)
        return False, str(exc)

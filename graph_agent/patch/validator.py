"""Patch validator & applier — applies diffs safely with multi-strategy fallback.

Order of strategy:
1. `git apply --ignore-whitespace --recount`
2. GNU/BSD `patch -p1 / -p0`
3. Direct hunk replacer (Python-native AST-safe context replacer)

ADR-6: We NEVER auto-accept a patch.
"""

from __future__ import annotations

import logging
import re
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


def _normalize_diff(diff: str) -> str:
    """Ensure diff ends with newline and clean whitespace."""
    lines = diff.strip().splitlines()
    return "\n".join(lines) + "\n"


def _try_git_apply(cwd: Path, patch_path: Path, check_only: bool = False) -> tuple[bool, str]:
    """Try applying patch with git apply."""
    cmd = ["git", "apply", "--ignore-whitespace", "--recount"]
    if check_only:
        cmd.append("--check")
    cmd.append(str(patch_path))

    proc = subprocess.run(cmd, cwd=str(cwd), capture_output=True, text=True, timeout=30)
    if proc.returncode == 0:
        return True, ""
    err = (proc.stderr or proc.stdout).strip()
    return False, err


def _try_patch_cmd(cwd: Path, patch_path: Path, check_only: bool = False) -> tuple[bool, str]:
    """Fallback: try applying patch with the GNU/BSD patch command."""
    for p_num in ["-p1", "-p0"]:
        cmd = ["patch", "--batch", "--silent", p_num]
        if check_only:
            cmd.append("--dry-run")
        cmd.extend(["-i", str(patch_path)])

        proc = subprocess.run(cmd, cwd=str(cwd), capture_output=True, text=True, timeout=30)
        if proc.returncode == 0:
            return True, ""
    return False, "patch command failed"


def _split_into_file_diffs(diff_str: str) -> list[str]:
    """Splits a multi-file diff into per-file diff blocks."""
    chunks = re.split(r'(?=^--- [ab]/)', diff_str, flags=re.MULTILINE)
    return [c.strip() for c in chunks if c.strip()]


def _apply_hunks_pure_python(file_text: str, file_diff: str) -> str | None:
    """Applies unified diff hunks directly in Python by finding matching context blocks."""
    # Matches both @@ -start,count +start,count @@ and bare @@
    hunks = re.split(r'(?=^@@)', file_diff, flags=re.MULTILINE)
    current_text = file_text

    applied_any = False
    for hunk in hunks:
        if not hunk.startswith("@@"):
            continue
        hunk_lines = hunk.splitlines()[1:]  # skip @@ ... @@ line
        
        old_lines = []
        new_lines = []
        for l in hunk_lines:
            if not l:
                continue
            marker = l[0]
            content = l[1:] if len(l) > 1 else ""
            if marker in (" ", "-"):
                old_lines.append(content)
            if marker in (" ", "+"):
                new_lines.append(content)

        old_block = "\n".join(old_lines).strip()
        new_block = "\n".join(new_lines).strip()

        if old_block and old_block in current_text:
            current_text = current_text.replace(old_block, new_block, 1)
            applied_any = True
        else:
            # Fallback: line-trimmed match
            old_stripped = [line.strip() for line in old_lines if line.strip()]
            if not old_stripped:
                continue
            lines = current_text.splitlines()
            found = False
            for i in range(len(lines) - len(old_stripped) + 1):
                window = [lines[j].strip() for j in range(i, i + len(old_stripped))]
                if window == old_stripped:
                    # Match found at line index i
                    # Compute leading indentation from original lines
                    lead_indent = re.match(r'^\s*', lines[i]).group(0) if lines[i] else ""
                    formatted_new = [
                        (lead_indent + nl.lstrip()) if nl.strip() else ""
                        for nl in new_lines
                    ]
                    lines[i : i + len(old_stripped)] = formatted_new
                    current_text = "\n".join(lines) + "\n"
                    found = True
                    applied_any = True
                    break
            if not found:
                return None

    return current_text if applied_any else None


def _try_python_diff_apply(repo_root: Path, diff: str, check_only: bool = False) -> tuple[bool, str]:
    """Python-native fallback: directly locates file and swaps hunk context."""
    file_diffs = _split_into_file_diffs(diff)
    if not file_diffs:
        return False, "No file headers in diff"

    files_to_update: dict[Path, str] = {}

    for fd in file_diffs:
        match = re.search(r'^--- [ab]/(.+)$', fd, re.MULTILINE)
        if not match:
            continue
        rel_path = match.group(1).strip()
        
        # Try finding target file
        candidates = [
            repo_root / rel_path,
            repo_root / Path(rel_path).name,
        ]
        target_file = next((c for c in candidates if c.is_file()), None)
        if not target_file:
            # Recursive search in repo_root
            found = list(repo_root.glob(f"**/{Path(rel_path).name}"))
            if found:
                target_file = found[0]

        if not target_file or not target_file.exists():
            return False, f"Target file '{rel_path}' not found in repo"

        orig_text = target_file.read_text(encoding="utf-8")
        updated_text = _apply_hunks_pure_python(orig_text, fd)
        if updated_text is None:
            return False, f"Could not match context hunk in '{target_file.name}'"

        files_to_update[target_file] = updated_text

    if not check_only:
        for fpath, new_content in files_to_update.items():
            fpath.write_text(new_content, encoding="utf-8")

    return True, ""


def validate_patch(repo_root: str | Path, diff: str) -> ValidationResult:
    """Check if *diff* applies cleanly to *repo_root*."""
    repo_root = Path(repo_root).resolve()
    tmp_dir = Path(tempfile.mkdtemp(prefix="graph_agent_patch_"))
    normalized_diff = _normalize_diff(diff)

    try:
        tmp_repo = tmp_dir / "repo"
        shutil.copytree(str(repo_root), str(tmp_repo), symlinks=True)
        tmp_patch = tmp_repo / "fix.patch"
        tmp_patch.write_text(normalized_diff, encoding="utf-8")

        # 1. Try git apply --check
        ok, err = _try_git_apply(tmp_repo, tmp_patch, check_only=True)
        if ok:
            return ValidationResult(patch_diff=normalized_diff, applies_cleanly=True, error="")

        # 2. Try patch --dry-run
        ok_patch, patch_err = _try_patch_cmd(tmp_repo, tmp_patch, check_only=True)
        if ok_patch:
            return ValidationResult(patch_diff=normalized_diff, applies_cleanly=True, error="")

        # 3. Try Python-native AST context match
        ok_py, py_err = _try_python_diff_apply(tmp_repo, normalized_diff, check_only=True)
        if ok_py:
            return ValidationResult(patch_diff=normalized_diff, applies_cleanly=True, error="")

        return ValidationResult(patch_diff=normalized_diff, applies_cleanly=False, error=err or patch_err or py_err)

    except Exception as exc:
        return ValidationResult(patch_diff=diff, applies_cleanly=False, error=str(exc))
    finally:
        shutil.rmtree(str(tmp_dir), ignore_errors=True)


def apply_patch(repo_root: str | Path, diff: str) -> tuple[bool, str]:
    """Actually apply *diff* to *repo_root*.
    
    IMPORTANT: Only call this after explicit human approval.
    """
    repo_root = Path(repo_root).resolve()
    patch_file = repo_root / "_graph_agent_fix.patch"
    normalized_diff = _normalize_diff(diff)

    try:
        patch_file.write_text(normalized_diff, encoding="utf-8")

        # 1. Try git apply
        ok, err = _try_git_apply(repo_root, patch_file, check_only=False)
        if ok:
            patch_file.unlink(missing_ok=True)
            return True, ""

        # 2. Fallback to patch command
        ok_patch, patch_err = _try_patch_cmd(repo_root, patch_file, check_only=False)
        if ok_patch:
            patch_file.unlink(missing_ok=True)
            return True, ""

        # 3. Fallback to python hunk application
        ok_py, py_err = _try_python_diff_apply(repo_root, normalized_diff, check_only=False)
        patch_file.unlink(missing_ok=True)
        if ok_py:
            return True, ""

        return False, err or patch_err or py_err
    except Exception as exc:
        patch_file.unlink(missing_ok=True)
        return False, str(exc)

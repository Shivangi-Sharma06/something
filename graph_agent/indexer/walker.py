"""Gitignore-aware recursive file walker.

Uses pathspec (bundled with gitpython) to honour .gitignore rules.
Falls back gracefully if no .gitignore is found.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterator

from graph_agent.settings import get_settings


def _load_gitignore_spec(repo_root: Path):
    """Return a pathspec.PathSpec from the repo's .gitignore, or None."""
    try:
        import pathspec  # bundled with gitpython
    except ImportError:
        return None

    gitignore_path = repo_root / ".gitignore"
    if not gitignore_path.exists():
        return None
    patterns = gitignore_path.read_text(encoding="utf-8").splitlines()
    return pathspec.PathSpec.from_lines("gitignore", patterns)


def walk_repo(repo_root: str | Path) -> Iterator[Path]:
    """Yield all indexable source files under *repo_root*.

    Respects:
    - .gitignore patterns (via pathspec / gitpython)
    - settings.indexer.skip_dirs  (always-skip directory names)
    - settings.indexer.extensions (only yield files with matching extensions)
    - settings.indexer.max_file_size (skip files that are too large)

    Files are yielded in deterministic order (sorted).
    """
    root = Path(repo_root).resolve()
    s = get_settings().indexer

    # Build a flat set of allowed extensions
    allowed_exts: set[str] = set()
    for exts in s.extensions.values():
        allowed_exts.update(exts)

    skip_dirs = set(s.skip_dirs)
    spec = _load_gitignore_spec(root)

    def _ignored(path: Path) -> bool:
        if spec is None:
            return False
        rel = str(path.relative_to(root))
        return spec.match_file(rel)

    collected: list[Path] = []

    for dirpath, dirnames, filenames in os.walk(root, topdown=True):
        current = Path(dirpath)

        # Prune directories in-place so os.walk won't descend into them
        dirnames[:] = sorted([
            d for d in dirnames
            if d not in skip_dirs and not _ignored(current / d)
        ])

        for fname in sorted(filenames):
            fpath = current / fname
            if fpath.suffix not in allowed_exts:
                continue
            if _ignored(fpath):
                continue
            try:
                size = fpath.stat().st_size
            except OSError:
                continue
            if size > s.max_file_size:
                continue
            collected.append(fpath)

    yield from collected

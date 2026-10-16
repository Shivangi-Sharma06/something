"""Unit tests for graph_agent.indexer.walker."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pytest

from graph_agent.indexer.walker import walk_repo


def _make_tree(base: Path, structure: dict) -> None:
    """Recursively create a directory tree from a dict spec."""
    for name, content in structure.items():
        path = base / name
        if isinstance(content, dict):
            path.mkdir(parents=True, exist_ok=True)
            _make_tree(path, content)
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content or "", encoding="utf-8")


class TestWalkRepo:
    def test_finds_python_files(self, tmp_path):
        _make_tree(tmp_path, {
            "main.py": "x = 1",
            "utils.py": "y = 2",
            "README.md": "# readme",
        })
        files = list(walk_repo(tmp_path))
        names = {f.name for f in files}
        assert "main.py" in names
        assert "utils.py" in names
        assert "README.md" not in names  # not a Python file

    def test_skips_venv_directory(self, tmp_path):
        _make_tree(tmp_path, {
            "main.py": "x = 1",
            ".venv": {"lib": {"site-packages": {"something.py": "pass"}}},
        })
        files = list(walk_repo(tmp_path))
        for f in files:
            assert ".venv" not in str(f)

    def test_skips_pycache(self, tmp_path):
        _make_tree(tmp_path, {
            "main.py": "x = 1",
            "__pycache__": {"main.cpython-312.pyc": ""},
        })
        files = list(walk_repo(tmp_path))
        for f in files:
            assert "__pycache__" not in str(f)

    def test_respects_gitignore(self, tmp_path):
        _make_tree(tmp_path, {
            ".gitignore": "ignored_module.py\n",
            "main.py": "x = 1",
            "ignored_module.py": "secret = True",
        })
        files = list(walk_repo(tmp_path))
        names = {f.name for f in files}
        assert "main.py" in names
        assert "ignored_module.py" not in names

    def test_skips_large_files(self, tmp_path):
        big = tmp_path / "big.py"
        big.write_bytes(b"x = 1\n" * 100000)  # ~600 KB
        small = tmp_path / "small.py"
        small.write_text("x = 1", encoding="utf-8")
        files = list(walk_repo(tmp_path))
        names = {f.name for f in files}
        assert "small.py" in names
        assert "big.py" not in names

    def test_output_is_sorted(self, tmp_path):
        _make_tree(tmp_path, {
            "z_last.py": "pass",
            "a_first.py": "pass",
            "m_middle.py": "pass",
        })
        files = list(walk_repo(tmp_path))
        names = [f.name for f in files]
        assert names == sorted(names)

    def test_empty_repo(self, tmp_path):
        files = list(walk_repo(tmp_path))
        assert files == []

    def test_sample_repo(self, sample_repo_path):
        """The fixture sample repo must have at least main.py and utils.py."""
        files = list(walk_repo(sample_repo_path))
        names = {f.name for f in files}
        assert "main.py" in names
        assert "utils.py" in names

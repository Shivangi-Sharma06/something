"""Unit tests for graph_agent.indexer.parser."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from graph_agent.indexer.parser import parse_file, detect_language, ParseResult


SAMPLE_REPO = Path(__file__).parent.parent / "fixtures" / "sample_repo"


class TestDetectLanguage:
    def test_python(self, tmp_path):
        assert detect_language(tmp_path / "foo.py") == "python"

    def test_js(self, tmp_path):
        assert detect_language(tmp_path / "foo.js") == "javascript"

    def test_unknown(self, tmp_path):
        assert detect_language(tmp_path / "foo.md") is None


class TestParseFile:
    def _write(self, tmp_path: Path, name: str, content: str) -> Path:
        p = tmp_path / name
        p.write_text(content, encoding="utf-8")
        return p

    def test_returns_none_for_unsupported(self, tmp_path):
        f = self._write(tmp_path, "readme.md", "# hello")
        result = parse_file(f, tmp_path)
        assert result is None

    def test_content_hash(self, tmp_path):
        src = "def foo(): pass\n"
        f = self._write(tmp_path, "a.py", src)
        result = parse_file(f, tmp_path)
        assert result is not None
        expected = hashlib.sha256(src.encode()).hexdigest()
        assert result.content_hash == expected

    def test_extracts_function(self, tmp_path):
        src = "def hello_world():\n    pass\n"
        f = self._write(tmp_path, "a.py", src)
        result = parse_file(f, tmp_path)
        assert result is not None
        names = [s.name for s in result.symbols]
        assert "hello_world" in names

    def test_extracts_class_and_method(self, tmp_path):
        src = (
            "class Foo:\n"
            "    def bar(self):\n"
            "        pass\n"
        )
        f = self._write(tmp_path, "a.py", src)
        result = parse_file(f, tmp_path)
        assert result is not None
        kinds = {s.kind for s in result.symbols}
        names = {s.name for s in result.symbols}
        assert "class" in kinds
        assert "method" in kinds
        assert "Foo" in names
        assert "bar" in names

    def test_qualified_name_includes_module(self, tmp_path):
        src = "def top_func():\n    pass\n"
        f = self._write(tmp_path, "mymod.py", src)
        result = parse_file(f, tmp_path)
        assert result is not None
        qnames = [s.qualified_name for s in result.symbols]
        assert "mymod.top_func" in qnames

    def test_nested_module_path(self, tmp_path):
        subdir = tmp_path / "pkg" / "sub"
        subdir.mkdir(parents=True)
        src = "def inner(): pass\n"
        f = subdir / "mod.py"
        f.write_text(src, encoding="utf-8")
        result = parse_file(f, tmp_path)
        assert result is not None
        qnames = [s.qualified_name for s in result.symbols]
        assert "pkg.sub.mod.inner" in qnames

    def test_extracts_import(self, tmp_path):
        src = "import os\nfrom pathlib import Path\n"
        f = self._write(tmp_path, "a.py", src)
        result = parse_file(f, tmp_path)
        assert result is not None
        assert any("os" in name for name in result.import_names)
        assert any("pathlib" in name for name in result.import_names)

    def test_extracts_call_references(self, tmp_path):
        src = (
            "def caller():\n"
            "    callee()\n"
            "\n"
            "def callee():\n"
            "    pass\n"
        )
        f = self._write(tmp_path, "a.py", src)
        result = parse_file(f, tmp_path)
        assert result is not None
        callers = [caller for caller, _ in result.raw_calls]
        callees = [callee for _, callee in result.raw_calls]
        assert any("caller" in c for c in callers)
        assert any("callee" in c for c in callees)

    def test_extracts_docstring(self, tmp_path):
        src = (
            'def documented():\n'
            '    """This is a docstring."""\n'
            '    pass\n'
        )
        f = self._write(tmp_path, "a.py", src)
        result = parse_file(f, tmp_path)
        assert result is not None
        docs = [s.docstring for s in result.symbols if s.name == "documented"]
        assert docs and "docstring" in docs[0]

    def test_sample_repo_utils(self):
        """Parse the fixture utils.py and verify known symbols."""
        f = SAMPLE_REPO / "utils.py"
        result = parse_file(f, SAMPLE_REPO)
        assert result is not None
        names = {s.name for s in result.symbols}
        assert {"add", "subtract", "divide", "format_result"}.issubset(names)

    def test_sample_repo_main(self):
        """Parse the fixture main.py and verify class + methods."""
        f = SAMPLE_REPO / "main.py"
        result = parse_file(f, SAMPLE_REPO)
        assert result is not None
        names = {s.name for s in result.symbols}
        assert "Calculator" in names
        assert "run_divide" in names

    def test_line_numbers(self, tmp_path):
        src = (
            "# header\n"
            "def foo():\n"
            "    pass\n"
        )
        f = self._write(tmp_path, "a.py", src)
        result = parse_file(f, tmp_path)
        assert result is not None
        foo = next(s for s in result.symbols if s.name == "foo")
        assert foo.line_start == 2

    def test_handles_unreadable_file(self, tmp_path):
        f = tmp_path / "missing.py"
        # Don't create the file — parse_file should handle OSError gracefully
        result = parse_file(f, tmp_path)
        assert result is not None
        assert result.errors  # should report an error, not crash

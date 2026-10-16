"""Tree-sitter based source code parser.

Extracts symbols (functions, classes, methods, imports) from Python files
using tree-sitter ASTs.  Returns plain dataclasses — no DB dependency here.

ADR: The parser is intentionally stateless and DB-free.  The ingester owns
the write path.  This makes the parser fast to unit-test in isolation.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import tree_sitter_python as tspython
from tree_sitter import Language, Parser, Node


# ---------------------------------------------------------------------------
# Language registry — easy to extend for JS, TS, etc.
# ---------------------------------------------------------------------------
_LANG_MAP: dict[str, Language] = {
    "python": Language(tspython.language()),
}


def get_language(name: str) -> Language:
    if name not in _LANG_MAP:
        raise ValueError(f"Unsupported language: {name!r}. Available: {list(_LANG_MAP)}")
    return _LANG_MAP[name]


def detect_language(path: Path) -> Optional[str]:
    ext_map = {
        ".py": "python",
        ".js": "javascript",
        ".mjs": "javascript",
        ".ts": "typescript",
    }
    return ext_map.get(path.suffix)


# ---------------------------------------------------------------------------
# Symbol dataclass
# ---------------------------------------------------------------------------
@dataclass
class SymbolInfo:
    kind: str           # function | class | method | import | parameter
    name: str
    qualified_name: str  # dotted path within the module
    parent_qualified_name: Optional[str] = None
    line_start: int = 0
    line_end: int = 0
    byte_start: int = 0
    byte_end: int = 0
    signature: Optional[str] = None
    docstring: Optional[str] = None


@dataclass
class ParseResult:
    language: str
    content_hash: str          # SHA-256 hex
    symbols: list[SymbolInfo] = field(default_factory=list)
    import_names: list[str] = field(default_factory=list)   # raw imported names/modules
    raw_calls: list[tuple[str, str]] = field(default_factory=list)  # (caller_qual_name, callee_name)
    errors: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _text(node: Node, src: bytes) -> str:
    return src[node.start_byte:node.end_byte].decode("utf-8", errors="replace")


def _first_string_node(node: Node) -> Optional[Node]:
    """Return the first string literal child (for docstrings)."""
    for child in node.children:
        if child.type in ("expression_statement",):
            for grandchild in child.children:
                if grandchild.type in ("string", "concatenated_string"):
                    return grandchild
    return None


def _get_docstring(body_node: Optional[Node], src: bytes) -> Optional[str]:
    if body_node is None:
        return None
    sn = _first_string_node(body_node)
    if sn is None:
        return None
    raw = _text(sn, src).strip()
    # Strip quotes
    for q in ('"""', "'''", '"', "'"):
        if raw.startswith(q) and raw.endswith(q) and len(raw) > 2 * len(q):
            return raw[len(q):-len(q)].strip()
    return raw


# ---------------------------------------------------------------------------
# Call extraction helpers
# ---------------------------------------------------------------------------
_CALL_QUERY_CACHE: dict[str, object] = {}


def _extract_calls_from_node(func_node: Node, src: bytes, lang: Language) -> list[str]:
    """Return a list of callee name strings called within *func_node*."""
    call_type = "call" if lang == _LANG_MAP.get("python") else "call_expression"
    calls: list[str] = []

    def _walk(node: Node):
        if node.type == "call":
            func_child = node.child_by_field_name("function")
            if func_child:
                raw = _text(func_child, src)
                # Take only the last segment for simple calls, keep full for attribute access
                calls.append(raw)
        for child in node.children:
            _walk(child)

    _walk(func_node)
    return calls


# ---------------------------------------------------------------------------
# Python parser
# ---------------------------------------------------------------------------
def _parse_python(src: bytes, module_qual: str) -> ParseResult:
    """Parse Python source and extract symbols + import + call info."""
    lang = _LANG_MAP["python"]
    parser = Parser(lang)
    tree = parser.parse(src)
    root = tree.root_node

    symbols: list[SymbolInfo] = []
    import_names: list[str] = []
    raw_calls: list[tuple[str, str]] = []
    errors: list[str] = []

    if root.has_error:
        errors.append("Tree-sitter reported parse errors")

    def _qual(*parts: str) -> str:
        return ".".join(p for p in parts if p)

    def _visit(node: Node, scope: str, class_scope: Optional[str] = None):
        """Recursively walk the AST, collecting symbols."""

        if node.type == "function_definition":
            name_node = node.child_by_field_name("name")
            if name_node is None:
                return
            name = _text(name_node, src)
            qual = _qual(scope, name)
            kind = "method" if class_scope else "function"
            body = node.child_by_field_name("body")
            doc = _get_docstring(body, src)

            # Signature: the full "def ..." first line
            sig_end = node.start_point[0]  # line of def keyword
            sig_lines = src.decode("utf-8", errors="replace").split("\n")
            sig = sig_lines[node.start_point[0]] if node.start_point[0] < len(sig_lines) else None

            sym = SymbolInfo(
                kind=kind,
                name=name,
                qualified_name=qual,
                parent_qualified_name=scope if scope != module_qual else None,
                line_start=node.start_point[0] + 1,
                line_end=node.end_point[0] + 1,
                byte_start=node.start_byte,
                byte_end=node.end_byte,
                signature=sig,
                docstring=doc,
            )
            symbols.append(sym)

            # Extract calls made from this function
            for callee_name in _extract_calls_from_node(node, src, lang):
                raw_calls.append((qual, callee_name))

            # Recurse into nested functions
            if body:
                for child in body.children:
                    _visit(child, qual, class_scope)

        elif node.type == "class_definition":
            name_node = node.child_by_field_name("name")
            if name_node is None:
                return
            name = _text(name_node, src)
            qual = _qual(scope, name)
            body = node.child_by_field_name("body")
            doc = _get_docstring(body, src)

            sym = SymbolInfo(
                kind="class",
                name=name,
                qualified_name=qual,
                parent_qualified_name=scope if scope != module_qual else None,
                line_start=node.start_point[0] + 1,
                line_end=node.end_point[0] + 1,
                byte_start=node.start_byte,
                byte_end=node.end_byte,
                docstring=doc,
            )
            symbols.append(sym)

            # Recurse into class body to find methods
            if body:
                for child in body.children:
                    _visit(child, qual, class_scope=qual)

        elif node.type == "import_statement":
            # e.g. import os, import sys
            for child in node.children:
                if child.type in ("dotted_name", "aliased_import"):
                    import_names.append(_text(child, src).split(" as ")[0].strip())

        elif node.type == "import_from_statement":
            # e.g. from pathlib import Path
            module_node = node.child_by_field_name("module_name")
            module_name = _text(module_node, src) if module_node else ""
            for child in node.children:
                if child.type == "dotted_name" and child != module_node:
                    import_names.append(f"{module_name}.{_text(child, src)}")
                elif child.type == "aliased_import":
                    import_names.append(f"{module_name}.{_text(child, src).split(' as ')[0].strip()}")
                elif child.type == "wildcard_import":
                    import_names.append(f"{module_name}.*")

        else:
            # Recurse into all other node types at top-level
            for child in node.children:
                _visit(child, scope, class_scope)

    for child in root.children:
        _visit(child, module_qual)

    content_hash = hashlib.sha256(src).hexdigest()
    return ParseResult(
        language="python",
        content_hash=content_hash,
        symbols=symbols,
        import_names=import_names,
        raw_calls=raw_calls,
        errors=errors,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def parse_file(path: Path, repo_root: Path) -> Optional[ParseResult]:
    """Parse *path* and return a ParseResult, or None if unsupported/unreadable."""
    lang = detect_language(path)
    if lang is None:
        return None
    if lang not in _LANG_MAP:
        return None

    try:
        src = path.read_bytes()
    except OSError as exc:
        return ParseResult(
            language=lang,
            content_hash="",
            errors=[f"Cannot read file: {exc}"],
        )

    # Build the module qualified name from the relative path
    try:
        rel = path.relative_to(repo_root)
    except ValueError:
        rel = path.name
    # Convert path to dotted module name: app/routers/auth.py → app.routers.auth
    parts = list(rel.parts)
    if parts[-1].endswith(".py"):
        parts[-1] = parts[-1][:-3]
    if parts[-1] == "__init__":
        parts = parts[:-1]
    module_qual = ".".join(parts)

    if lang == "python":
        return _parse_python(src, module_qual)

    # Placeholder for future languages
    return ParseResult(language=lang, content_hash=hashlib.sha256(src).hexdigest())

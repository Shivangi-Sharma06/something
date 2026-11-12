"""Permanent AST Tree module — constructs and visualizes the repository AST graph.

Extracts the full hierarchical structural representation of the repository from PostgreSQL:
  Repository → Files → Classes/Modules → Methods/Functions
Along with AST metadata (signatures, line spans, docstrings) and call graph relationships
(CONTAINS, CALLS edges).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from rich.tree import Tree
from sqlalchemy.orm import Session
from sqlalchemy import text

from graph_agent.db.models import Repository, File, Symbol, Edge

log = logging.getLogger(__name__)


@dataclass
class ASTSymbolNode:
    symbol_id: int
    name: str
    qualified_name: str
    kind: str
    signature: str | None
    line_start: int | None
    line_end: int | None
    docstring: str | None
    children: list[ASTSymbolNode] = field(default_factory=list)
    callers: list[str] = field(default_factory=list)
    callees: list[str] = field(default_factory=list)


@dataclass
class ASTFileNode:
    file_id: int
    rel_path: str
    language: str | None
    size_bytes: int | None
    symbols: list[ASTSymbolNode] = field(default_factory=list)


@dataclass
class PermanentASTTree:
    repo_id: int
    repo_name: str
    root_path: str
    files: list[ASTFileNode] = field(default_factory=list)
    total_files: int = 0
    total_symbols: int = 0
    total_edges: int = 0

    def to_rich_tree(self) -> Tree:
        """Convert the permanent AST structure into a Rich Tree for terminal display."""
        root = Tree(
            f"[bold green]Permanent AST Tree[/bold green]: "
            f"[cyan]{self.repo_name}[/cyan] ({self.total_files} files, {self.total_symbols} symbols, {self.total_edges} edges)"
        )

        for file_node in self.files:
            lang_label = f" ({file_node.language})" if file_node.language else ""
            file_tree = root.add(f"[bold blue][file] {file_node.rel_path}[/bold blue]{lang_label}")

            if not file_node.symbols:
                file_tree.add("[dim italic](no indexed top-level symbols)[/dim italic]")
                continue

            for sym in file_node.symbols:
                _add_symbol_to_tree(file_tree, sym)

        return root


def _add_symbol_to_tree(parent_tree: Tree, sym: ASTSymbolNode) -> None:
    lines = f"L{sym.line_start}-L{sym.line_end}" if sym.line_start else ""
    kind_style = "magenta" if sym.kind == "class" else "yellow" if sym.kind in ("function", "method") else "cyan"
    tag = f"[{sym.kind}]"
    
    label = f"[{kind_style}]{tag}[/{kind_style}] [bold]{sym.name}[/bold]"
    if sym.signature:
        sig = sym.signature.split("\n")[0].strip()
        if len(sig) > 60:
            sig = sig[:57] + "..."
        label += f" [dim]{sig}[/dim]"
    if lines:
        label += f" [dim]({lines})[/dim]"

    node = parent_tree.add(label)

    if sym.callers:
        callers_str = ", ".join(sym.callers[:3])
        if len(sym.callers) > 3:
            callers_str += f" (+{len(sym.callers) - 3} more)"
        node.add(f"[dim]<- Called by: {callers_str}[/dim]")

    if sym.callees:
        callees_str = ", ".join(sym.callees[:3])
        if len(sym.callees) > 3:
            callees_str += f" (+{len(sym.callees) - 3} more)"
        node.add(f"[dim]-> Calls: {callees_str}[/dim]")

    for child in sym.children:
        _add_symbol_to_tree(node, child)


def build_permanent_ast_tree(session: Session, repo_id: int) -> PermanentASTTree:
    """Load and construct the permanent AST Tree data structure for a repository."""
    repo = session.query(Repository).filter_by(id=repo_id).first()
    if not repo:
        raise ValueError(f"Repository with id {repo_id} not found.")

    files = session.query(File).filter_by(repo_id=repo_id).order_by(File.rel_path).all()
    symbols = session.query(Symbol).filter_by(repo_id=repo_id).all()
    edges = session.query(Edge).filter_by(repo_id=repo_id).all()

    # Build symbol map
    sym_map: dict[int, ASTSymbolNode] = {}
    for s in symbols:
        sym_map[s.id] = ASTSymbolNode(
            symbol_id=s.id,
            name=s.name,
            qualified_name=s.qualified_name,
            kind=s.kind,
            signature=s.signature,
            line_start=s.line_start,
            line_end=s.line_end,
            docstring=s.docstring,
        )

    # Attach CALLS edges
    for e in edges:
        if e.edge_type == "CALLS" and e.source_id in sym_map and e.target_id in sym_map:
            src = sym_map[e.source_id]
            tgt = sym_map[e.target_id]
            src.callees.append(tgt.name)
            tgt.callers.append(src.name)

    # Attach children hierarchy (CONTAINS or parent_id)
    child_ids = set()
    for s in symbols:
        if s.parent_id and s.parent_id in sym_map and s.id in sym_map:
            sym_map[s.parent_id].children.append(sym_map[s.id])
            child_ids.add(s.id)

    # Map symbols to files (only top-level symbols, children are nested)
    file_nodes: list[ASTFileNode] = []
    total_syms = len(symbols)

    for f in files:
        file_syms = [
            sym_map[s.id]
            for s in symbols
            if s.file_id == f.id and s.id not in child_ids and s.id in sym_map
        ]
        file_nodes.append(ASTFileNode(
            file_id=f.id,
            rel_path=f.rel_path,
            language=f.language,
            size_bytes=f.size_bytes,
            symbols=file_syms,
        ))

    return PermanentASTTree(
        repo_id=repo.id,
        repo_name=repo.name,
        root_path=repo.root_path,
        files=file_nodes,
        total_files=len(files),
        total_symbols=total_syms,
        total_edges=len(edges),
    )

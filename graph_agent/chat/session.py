"""Interactive Chat and Decision Graph module for Graph-Based Coding Agent.

Workflow:
1. One-time permanent AST repository graph construction & persistence in PostgreSQL.
2. User provides prompt/task in an interactive chat session.
3. Localizer & Graph Traversal extract a temporary Decision Subgraph:
   - Reasons why specific files & symbols are considered
   - Shows callers, callees, and exact line ranges targeted
4. LLM proposes a minimal unified diff patch with explicit step-by-step reasoning.
5. User reviews the explanation and the diff.
6. Upon user confirmation (approve / y), the patch is applied directly to the repo.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.tree import Tree
from rich.prompt import Prompt, Confirm
from rich.syntax import Syntax

from graph_agent.db.connection import get_session
from graph_agent.db.models import Repository, File, Symbol, Edge
from graph_agent.graph.builder import build_graph
from graph_agent.indexer.ingester import index_repository
from graph_agent.localizer.ranker import localize, ScoredSymbol
from graph_agent.graph.queries import get_callers, get_callees
from graph_agent.context.selector import BudgetContextSelector
from graph_agent.llm.base import ModelAdapter
from graph_agent.llm.openai_compat import get_adapter
from graph_agent.patch.generator import generate_patch
from graph_agent.patch.validator import validate_patch, apply_patch
from graph_agent.settings import get_settings

console = Console()
log = logging.getLogger(__name__)


@dataclass
class DecisionNode:
    symbol_id: int
    qualified_name: str
    kind: str
    rel_path: str
    line_start: int | None
    line_end: int | None
    score: float
    reason: str
    evidence_kind: str
    callers: list[dict[str, Any]] = field(default_factory=list)
    callees: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class DecisionGraph:
    prompt: str
    repo_root: str
    target_files: set[str]
    nodes: list[DecisionNode]
    summary_explanation: str


def ensure_permanent_graph(repo_path: str | Path, name: str | None = None) -> int:
    """Ensures the repository is parsed once with Tree-sitter and stored in PostgreSQL.
    
    If not already indexed, parses all files, stores AST symbols, and resolves edges.
    Returns repo_id.
    """
    repo_root = str(Path(repo_path).resolve())
    with get_session() as session:
        repo = session.query(Repository).filter_by(root_path=repo_root).first()
        if repo is None:
            console.print(f"[cyan]Indexing repository at {repo_root} (Tree-sitter AST)...[/cyan]")
            result = index_repository(session, repo_root, name=name or Path(repo_root).name)
            repo = session.query(Repository).filter_by(root_path=repo_root).first()
            console.print(f"[green]✓ Indexed {result.files_indexed} files, {result.symbols_upserted} symbols in PostgreSQL.[/green]")
        
        # Ensure graph edges are built
        console.print("[cyan]Building/updating permanent code graph edges...[/cyan]")
        g_result = build_graph(session, repo.id)
        console.print(
            f"[green]✓ Code graph ready: {g_result.contains_edges} CONTAINS edges, "
            f"{g_result.edges_resolved} CALLS resolved.[/green]"
        )
        return repo.id


def build_temporary_decision_graph(
    session,
    repo_id: int,
    user_prompt: str,
    top_k: int = 5,
) -> DecisionGraph:
    """Extracts a temporary decision subgraph answering WHY specific files and symbols

    were chosen based on user prompt and graph connectivity.
    """
    repo = session.query(Repository).filter_by(id=repo_id).first()
    repo_root = repo.root_path if repo else ""

    # Parse potential stack trace or test references inside the prompt
    candidates: list[ScoredSymbol] = localize(
        session,
        repo_id,
        description=user_prompt,
        stack_trace=user_prompt if "Traceback" in user_prompt or "line " in user_prompt else "",
        failing_test=user_prompt if "test_" in user_prompt else "",
        top_k=top_k,
    )

    decision_nodes: list[DecisionNode] = []
    target_files: set[str] = set()

    for cand in candidates:
        sym = session.query(Symbol).filter_by(id=cand.symbol_id).first()
        if not sym:
            continue
        
        file_obj = session.query(File).filter_by(id=sym.file_id).first()
        rel_path = file_obj.rel_path if file_obj else "unknown"
        target_files.add(rel_path)

        # Callers & callees
        callers = get_callers(session, repo_id, sym.id, max_depth=1)
        callees = get_callees(session, repo_id, sym.id, max_depth=1)

        reason = (
            f"Matched via {cand.evidence_source} ({cand.evidence_kind.value}) "
            f"with confidence score {cand.score:.2f}."
        )
        if callers:
            reason += f" Called by: {', '.join(c['qualified_name'] for c in callers[:2])}."
        if callees:
            reason += f" Calls: {', '.join(c['qualified_name'] for c in callees[:2])}."

        decision_nodes.append(DecisionNode(
            symbol_id=sym.id,
            qualified_name=cand.qualified_name,
            kind=sym.kind,
            rel_path=rel_path,
            line_start=sym.line_start,
            line_end=sym.line_end,
            score=cand.score,
            reason=reason,
            evidence_kind=cand.evidence_kind.value,
            callers=callers,
            callees=callees,
        ))

    explanation = (
        f"Selected {len(target_files)} file(s) ({', '.join(target_files)}) "
        f"and {len(decision_nodes)} target symbol(s) based on structural caller/callee traversal."
    )

    return DecisionGraph(
        prompt=user_prompt,
        repo_root=repo_root,
        target_files=target_files,
        nodes=decision_nodes,
        summary_explanation=explanation,
    )


def display_decision_graph(dg: DecisionGraph) -> None:
    """Renders the decision graph visually to the user in the terminal."""
    tree = Tree(f"[bold magenta]Temporary Decision Graph[/bold magenta] (Targeting {len(dg.target_files)} file(s))")

    for f in dg.target_files:
        file_node = tree.add(f"[bold cyan]File: {f}[/bold cyan]")
        for n in dg.nodes:
            if n.rel_path == f:
                lines = f"L{n.line_start}-L{n.line_end}" if n.line_start else "whole file"
                sym_label = f"[green]{n.qualified_name}[/green] ({n.kind}, {lines}) [Score: {n.score:.2f}]"
                sym_node = file_node.add(sym_label)
                sym_node.add(f"[dim]Rationale: {n.reason}[/dim]")
                if n.callers:
                    callers_label = sym_node.add("[yellow]Callers (who calls this):[/yellow]")
                    for c in n.callers:
                        callers_label.add(f"<- {c['qualified_name']}")
                if n.callees:
                    callees_label = sym_node.add("[yellow]Callees (what this calls):[/yellow]")
                    for c in n.callees:
                        callees_label.add(f"-> {c['qualified_name']}")

    console.print(tree)
    console.print(Panel(dg.summary_explanation, title="Decision Rationale", border_style="blue"))


def run_chat_session(
    repo_path: str | Path,
    adapter_name: str | None = None,
    model_name: str | None = None,
    api_base: str | None = None,
    api_key: str | None = None,
    interactive: bool = True,
) -> None:
    """Runs an interactive chat loop with the LLM to inspect, localize, generate,

    justify, and apply patches with human approval.
    """
    repo_path = Path(repo_path).resolve()
    console.print(Panel.fit(
        f"[bold green]Graph-Based Coding Agent Interactive Chat[/bold green]\n"
        f"Repository: [cyan]{repo_path}[/cyan]\n"
        f"Commands: Type your change request / prompt. Type [bold yellow]exit[/bold yellow] or [bold yellow]quit[/bold yellow] to leave.",
        border_style="green",
    ))

    # Phase 1: Permanent AST graph
    repo_id = ensure_permanent_graph(repo_path)
    s = get_settings()
    llm: ModelAdapter = get_adapter(
        adapter_name=adapter_name,
        model=model_name,
        base_url=api_base,
        api_key=api_key,
    )

    console.print(f"[bold cyan]LLM adapter connected:[/bold cyan] {llm.name()}\n")

    while True:
        try:
            user_prompt = Prompt.ask("\n[bold blue]Prompt / Request[/bold blue]").strip()
        except (KeyboardInterrupt, EOFError):
            console.print("\n[yellow]Session ended.[/yellow]")
            break

        if not user_prompt:
            continue
        if user_prompt.lower() in ("exit", "quit", "q"):
            console.print("[yellow]Exiting chat session.[/yellow]")
            break

        console.print("\n[cyan]Analyzing request & querying structural code graph...[/cyan]")

        with get_session() as session:
            # Phase 2: Temporary decision graph
            decision_graph = build_temporary_decision_graph(session, repo_id, user_prompt, top_k=5)
            display_decision_graph(decision_graph)

            if not decision_graph.nodes:
                console.print("[yellow]No relevant symbols found. Try refining your prompt or referencing function/file names.[/yellow]")
                continue

            # Phase 3: Build context & call LLM
            candidates = localize(session, repo_id, description=user_prompt, top_k=5)
            context = BudgetContextSelector().select(session, repo_id, candidates, s.context.token_budget)

        console.print("\n[cyan]Generating patch with reasoning from the decision graph...[/cyan]")
        patch = generate_patch(
            adapter=llm,
            description=user_prompt,
            stack_trace=user_prompt if "Traceback" in user_prompt else "",
            failing_test=user_prompt if "test_" in user_prompt else "",
            candidates=candidates,
            context=context,
        )

        if not patch.is_valid or not patch.diff.strip():
            console.print("[red]LLM could not formulate a valid unified diff.[/red]")
            if patch.root_cause:
                console.print(Panel(patch.root_cause, title="LLM Response", border_style="red"))
            continue

        # Display Root Cause / Reasoning
        console.print(Panel(
            patch.root_cause.strip(),
            title="[bold green]What & Why: LLM Rationale and Summary[/bold green]",
            border_style="green",
        ))

        # Display Diff
        syntax = Syntax(patch.diff, "diff", theme="monokai", line_numbers=True)
        console.print(Panel(syntax, title="Proposed Patch", border_style="cyan"))

        # Pre-validate patch cleanly in isolation
        val = validate_patch(repo_path, patch.diff)
        if val.applies_cleanly:
            console.print("[bold green]✓ Pre-validation check: Patch applies cleanly to the codebase.[/bold green]")
        else:
            console.print(f"[bold yellow]⚠ Pre-validation note: {val.error}[/bold yellow]")

        # Phase 4: Human Approval
        user_choice = Prompt.ask(
            "\n[bold yellow]Do you approve applying this patch to your repository? (y/yes to apply, n/no to discard)[/bold yellow]",
            default="n",
        ).strip().lower()

        approved = user_choice in ("y", "yes", "true", "1")

        if approved:
            applied, msg = apply_patch(repo_path, patch.diff)
            if applied:
                console.print(f"[bold green]✔ Patch applied successfully to {repo_path}![/bold green]")
                console.print("[dim]Re-indexing updated file(s) into graph...[/dim]")
                with get_session() as session:
                    index_repository(session, str(repo_path))
                    build_graph(session, repo_id)
                console.print("[green]✓ Graph synchronized with applied changes.[/green]")
            else:
                console.print(f"[red]Failed to apply patch: {msg}[/red]")
        else:
            console.print("[yellow]Patch discarded. Repository was not modified.[/yellow]")

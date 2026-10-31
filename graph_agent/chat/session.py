"""Interactive Chat and Decision Graph module for Graph-Based Coding Agent.

Workflow:
1. One-time permanent AST repository graph construction & persistence in PostgreSQL.
2. User provides prompt/task in an interactive chat session (or types /tree, /ast to view permanent AST).
3. System builds a Temporary Decision Tree:
   - Reasons why specific files & symbols are considered
   - Shows caller/callee links, confidence scores, and line ranges targeted
4. Real-time prompt & repository context rendering.
5. Real-time streaming response from the LLM.
6. For questions/inquiries: delivers full architectural answer.
   For bug fixes/refactors: provides step-by-step reasoning + unified diff, pre-validates cleanly,
   and upon user confirmation (y/yes), applies patch directly to the repository.
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.tree import Tree
from rich.prompt import Prompt
from rich.syntax import Syntax
from rich.live import Live
from rich.text import Text

from graph_agent.db.connection import get_session
from graph_agent.db.models import Repository
from graph_agent.graph.builder import build_graph
from graph_agent.graph.ast_tree import build_permanent_ast_tree, PermanentASTTree
from graph_agent.graph.decision_tree import build_temporary_decision_tree, TemporaryDecisionTree
from graph_agent.indexer.ingester import index_repository
from graph_agent.llm.base import ModelAdapter
from graph_agent.llm.openai_compat import get_adapter
from graph_agent.patch.generator import generate_patch, generate_answer, extract_patch
from graph_agent.patch.validator import validate_patch, apply_patch
from graph_agent.llm.ollama import select_ollama_model
from graph_agent.settings import get_settings

console = Console()
log = logging.getLogger(__name__)


def ensure_permanent_graph(repo_path: str | Path, name: str | None = None) -> tuple[int, PermanentASTTree]:
    """Ensures the repository is parsed with Tree-sitter and stored in PostgreSQL.
    
    If not already indexed, parses all files, stores AST symbols, and resolves edges.
    Returns (repo_id, ast_tree).
    """
    repo_root = str(Path(repo_path).resolve())
    with get_session() as session:
        repo = session.query(Repository).filter_by(root_path=repo_root).first()
        if repo is None:
            console.print(f"[cyan]Indexing repository at {repo_root} (Tree-sitter AST)...[/cyan]")
            result = index_repository(session, repo_root, repo_name=name or Path(repo_root).name)
            repo = session.query(Repository).filter_by(root_path=repo_root).first()
            console.print(f"[green]✓ Indexed {result.files_indexed} files, {result.symbols_upserted} symbols in PostgreSQL.[/green]")
        
        # Ensure graph edges are built
        console.print("[cyan]Building/updating permanent code graph edges...[/cyan]")
        g_result = build_graph(session, repo.id)
        console.print(
            f"[green]✓ Code graph ready: {g_result.contains_edges} CONTAINS edges, "
            f"{g_result.edges_resolved} CALLS resolved.[/green]"
        )

        ast_tree = build_permanent_ast_tree(session, repo.id)
        return repo.id, ast_tree


def display_prompt_and_context(prompt: str, dt: TemporaryDecisionTree, context: Any) -> None:
    """Renders the constructed prompt and repository context in real time before calling the LLM."""
    slices_desc = []
    if hasattr(context, "slices") and context.slices:
        for slc in context.slices:
            kind_lbl = "full source" if slc.is_full else "signature"
            slices_desc.append(f"• [cyan]{slc.rel_path}[/cyan] ({slc.qualified_name}, L{slc.line_start}-L{slc.line_end}) [{kind_lbl}]")
    else:
        slices_desc.append("• [dim]Repository structural summary / symbol scope[/dim]")

    context_text = "\n".join(slices_desc)
    panel_content = (
        f"[bold yellow]Request:[/bold yellow] {prompt}\n\n"
        f"[bold green]Repository Context Injected ({dt.tokens_used} est. tokens):[/bold green]\n"
        f"{context_text}"
    )
    console.print(Panel(panel_content, title="[bold]Real-Time Prompt & Context Preview[/bold]", border_style="blue"))


def run_chat_session(
    repo_path: str | Path,
    adapter_name: str | None = None,
    model_name: str | None = None,
    api_base: str | None = None,
    api_key: str | None = None,
    interactive: bool = True,
) -> None:
    """Runs an interactive chat loop with real-time prompt rendering, AST & decision trees,

    and streaming LLM responses.
    """
    repo_path = Path(repo_path).resolve()
    s = get_settings()
    effective_adapter = adapter_name or os.environ.get("GRAPH_AGENT_LLM_ADAPTER") or s.llm.adapter

    # If using local ollama and model not explicitly specified, prompt user with interactive dropdown
    if effective_adapter == "ollama" and not model_name and interactive:
        model_name = select_ollama_model(default_model="gptoss:20b", host_url=api_base, interactive=True)
    elif not model_name:
        model_name = os.environ.get("OLLAMA_MODEL") or os.environ.get("OPENAI_MODEL") or s.llm.model

    console.print(Panel.fit(
        f"[bold green]Graph-Based Coding Agent Interactive Chat[/bold green]\n"
        f"Repository: [cyan]{repo_path}[/cyan]\n"
        f"Model: [bold yellow]{model_name}[/bold yellow] via [cyan]{effective_adapter}[/cyan]\n"
        f"Commands:\n"
        f"  • Type your prompt/request.\n"
        f"  • Type [bold yellow]/model[/bold yellow] to switch local Ollama model.\n"
        f"  • Type [bold yellow]/tree[/bold yellow] to view permanent AST tree.\n"
        f"  • Type [bold yellow]exit[/bold yellow] to quit.",
        border_style="green",
    ))

    # Phase 1: Permanent AST graph
    repo_id, ast_tree = ensure_permanent_graph(repo_path)
    llm: ModelAdapter = get_adapter(
        adapter_name=effective_adapter,
        model=model_name,
        base_url=api_base,
        api_key=api_key,
    )

    console.print(f"[bold cyan]LLM adapter connected:[/bold cyan] {llm.name()}")
    console.print(
        f"[dim]Permanent AST indexed: {ast_tree.total_files} files, {ast_tree.total_symbols} symbols, "
        f"{ast_tree.total_edges} edges in PostgreSQL.[/dim]\n"
    )

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

        # User wants to switch Ollama model
        if user_prompt.lower() in ("/model", "/models", "/switch", "model"):
            new_model = select_ollama_model(default_model=model_name or "gptoss:20b", host_url=api_base, interactive=True)
            if new_model:
                model_name = new_model
                llm = get_adapter(
                    adapter_name=effective_adapter,
                    model=model_name,
                    base_url=api_base,
                    api_key=api_key,
                )
                console.print(f"[bold green]✓ Active model switched to:[/bold green] [bold cyan]{model_name}[/bold cyan] ({llm.name()})\n")
            continue

        # User wants to inspect permanent AST tree
        if user_prompt.lower() in ("/tree", "/ast", "tree", "show tree"):
            with get_session() as session:
                tree = build_permanent_ast_tree(session, repo_id)
                console.print(tree.to_rich_tree())
            continue

        console.print("\n[cyan]Analyzing request & querying structural code graph...[/cyan]")

        with get_session() as session:
            # Phase 2: Build Temporary Decision Tree
            decision_tree, context, candidates = build_temporary_decision_tree(
                session, repo_id, user_prompt, top_k=5
            )

            # Display the Temporary Decision Tree
            console.print(decision_tree.to_rich_tree())

            # Phase 3: Real-time Prompt & Repository Context Preview
            display_prompt_and_context(user_prompt, decision_tree, context)

        # Phase 4: Stream response in real time
        console.print("\n[bold cyan]LLM Response (Streaming in real time):[/bold cyan]")
        console.print("─" * 60)

        chunks: list[str] = []

        def _stream_cb(chunk: str) -> None:
            chunks.append(chunk)
            sys.stdout.write(chunk)
            sys.stdout.flush()

        try:
            if decision_tree.intent in ("repair", "refactor"):
                patch = generate_patch(
                    adapter=llm,
                    description=user_prompt,
                    stack_trace=user_prompt if "Traceback" in user_prompt else "",
                    failing_test=user_prompt if "test_" in user_prompt else "",
                    candidates=candidates,
                    context=context,
                    decision_tree_reasoning=decision_tree.to_prompt_context(),
                    stream_chunk_cb=_stream_cb,
                )
            else:
                full_text = generate_answer(
                    adapter=llm,
                    description=user_prompt,
                    context=context,
                    decision_tree_reasoning=decision_tree.to_prompt_context(),
                    stream_chunk_cb=_stream_cb,
                )
                patch = extract_patch(full_text)
        except Exception as exc:
            console.print(f"\n[red]Error generating response: {exc}[/red]")
            continue

        console.print("\n" + "─" * 60)

        # Phase 5: If a code patch was generated, validate and offer application
        if patch and patch.is_valid and patch.diff.strip():
            # Display formatted patch diff
            syntax = Syntax(patch.diff, "diff", theme="monokai", line_numbers=True)
            console.print(Panel(syntax, title="[bold cyan]Proposed Patch Diff[/bold cyan]", border_style="cyan"))

            # Pre-validate patch
            val = validate_patch(repo_path, patch.diff)
            if val.applies_cleanly:
                console.print("[bold green]✓ Pre-validation check: Patch applies cleanly to the repository.[/bold green]")
            else:
                console.print(f"[bold yellow]⚠ Pre-validation note: {val.error}[/bold yellow]")

            # Ask user confirmation
            user_choice = Prompt.ask(
                "\n[bold yellow]Apply this patch to your repository? (y/yes to apply, n/no to discard)[/bold yellow]",
                default="n",
            ).strip().lower()

            if user_choice in ("y", "yes", "true", "1"):
                applied, msg = apply_patch(repo_path, patch.diff)
                if applied:
                    console.print(f"[bold green]✔ Patch applied successfully to {repo_path}![/bold green]")
                    console.print("[dim]Updating permanent AST graph in PostgreSQL...[/dim]")
                    with get_session() as session:
                        index_repository(session, str(repo_path))
                        build_graph(session, repo_id)
                    console.print("[green]✓ Permanent AST graph synchronized.[/green]")
                else:
                    console.print(f"[red]Failed to apply patch: {msg}[/red]")
            else:
                console.print("[yellow]Patch discarded. Repository was not modified.[/yellow]")

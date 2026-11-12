"""graph-agent CLI entry point.

All commands share the same Click group.  Sub-commands for later phases
(localize, repair, verify, evaluate) will be added incrementally.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import click
from rich.console import Console
from rich.table import Table

console = Console()


def _setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


@click.group()
@click.option("--log-level", default="INFO", show_default=True,
              type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR"], case_sensitive=False))
def cli(log_level: str) -> None:
    """Graph-Based Repository-Level Coding Agent."""
    _setup_logging(log_level)


# ---------------------------------------------------------------------------
# graph-agent index
# ---------------------------------------------------------------------------
@cli.command("index")
@click.argument("repo_path", type=click.Path(exists=True, file_okay=False))
@click.option("--name", default=None, help="Repository display name (defaults to directory name)")
@click.option("--db-url", default=None, envvar="DATABASE_URL",
              help="PostgreSQL connection URL (overrides config)")
def index_cmd(repo_path: str, name: str | None, db_url: str | None) -> None:
    """Index a repository: parse source files and populate the graph database."""
    from graph_agent.db.connection import get_engine, get_session, override_engine
    from graph_agent.indexer.ingester import index_repository
    from graph_agent.settings import get_settings
    import sqlalchemy

    s = get_settings()
    url = db_url or s.database.url
    if db_url:
        eng = sqlalchemy.create_engine(url, future=True)
        override_engine(eng)

    console.print(f"[bold cyan]Indexing[/bold cyan] {repo_path!r} ...")

    try:
        with get_session() as session:
            result = index_repository(session, repo_path, repo_name=name)
    except Exception as exc:
        console.print(f"[red]ERROR:[/red] {exc}")
        sys.exit(1)

    # Pretty summary table
    t = Table(title="Indexing Summary", show_header=True)
    t.add_column("Metric", style="bold")
    t.add_column("Count", justify="right")
    t.add_row("Files visited",   str(result.files_visited))
    t.add_row("Files indexed",   str(result.files_indexed))
    t.add_row("Files skipped",   str(result.files_skipped))
    t.add_row("Symbols upserted", str(result.symbols_upserted))
    t.add_row("Errors",          str(len(result.errors)))
    console.print(t)

    if result.errors:
        console.print("[yellow]Errors:[/yellow]")
        for err in result.errors:
            console.print(f"  * {err}")

    console.print(f"[green]Done.[/green] Repo id={result.repo_id}")


# ---------------------------------------------------------------------------
# graph-agent db-check
# ---------------------------------------------------------------------------
@cli.command("db-check")
def db_check_cmd() -> None:
    """Verify database connectivity and show table row counts."""
    from graph_agent.db.connection import check_connection, get_engine
    from sqlalchemy import text

    if not check_connection():
        console.print("[red]Cannot connect to database.[/red]")
        sys.exit(1)

    console.print("[green]Database connection OK[/green]")

    tables = [
        "graph_agent.repositories",
        "graph_agent.files",
        "graph_agent.symbols",
        "graph_agent.edges",
        "graph_agent.unresolved_refs",
        "graph_agent.bug_reports",
        "graph_agent.repair_attempts",
    ]
    t = Table(title="Row Counts")
    t.add_column("Table", style="bold")
    t.add_column("Rows", justify="right")

    with get_engine().connect() as conn:
        for tbl in tables:
            count = conn.execute(text(f"SELECT COUNT(*) FROM {tbl}")).scalar()
            t.add_row(tbl, str(count))

    console.print(t)


# ---------------------------------------------------------------------------
# graph-agent build-graph
# ---------------------------------------------------------------------------
@cli.command("build-graph")
@click.argument("repo_path", type=click.Path(exists=True, file_okay=False))
@click.option("--repo-id", default=None, type=int,
              help="Override repo id (auto-detected from path if omitted)")
def build_graph_cmd(repo_path: str, repo_id: int | None) -> None:
    """Build or rebuild the code graph for an already-indexed repository."""
    from graph_agent.db.connection import get_session
    from graph_agent.db.models import Repository
    from graph_agent.graph.builder import build_graph

    with get_session() as session:
        if repo_id is None:
            from pathlib import Path
            root = str(Path(repo_path).resolve())
            repo = session.query(Repository).filter_by(root_path=root).first()
            if repo is None:
                console.print(f"[red]Repo not indexed yet. Run `graph-agent index {repo_path}` first.[/red]")
                import sys; sys.exit(1)
            repo_id = repo.id

        result = build_graph(session, repo_id)

    t = Table(title="Graph Build Summary", show_header=True)
    t.add_column("Metric", style="bold")
    t.add_column("Count", justify="right")
    t.add_row("CONTAINS edges",     str(result.contains_edges))
    t.add_row("CALLS resolved",     str(result.edges_resolved))
    t.add_row("CALLS unresolved",   str(result.edges_unresolved))
    t.add_row("Total edges created", str(result.edges_created))
    console.print(t)


# ---------------------------------------------------------------------------
# graph-agent tree (Permanent AST Tree)
# ---------------------------------------------------------------------------
@cli.command("tree")
@click.argument("repo_path", type=click.Path(exists=True, file_okay=False))
def tree_cmd(repo_path: str) -> None:
    """Display the permanent hierarchical AST tree of a repository."""
    from pathlib import Path
    from graph_agent.db.connection import get_session
    from graph_agent.db.models import Repository
    from graph_agent.indexer.ingester import index_repository
    from graph_agent.graph.builder import build_graph
    from graph_agent.graph.ast_tree import build_permanent_ast_tree

    root = str(Path(repo_path).resolve())
    with get_session() as session:
        repo = session.query(Repository).filter_by(root_path=root).first()
        if repo is None:
            console.print(f"[cyan]Repo not yet indexed. Indexing {root}...[/cyan]")
            index_repository(session, root)
            repo = session.query(Repository).filter_by(root_path=root).first()
            build_graph(session, repo.id)

        ast_tree = build_permanent_ast_tree(session, repo.id)
        console.print(ast_tree.to_rich_tree())


# ---------------------------------------------------------------------------
# graph-agent decision-tree (Temporary Decision Tree)
# ---------------------------------------------------------------------------
@cli.command("decision-tree")
@click.argument("repo_path", type=click.Path(exists=True, file_okay=False))
@click.option("--prompt", default="", help="Prompt or task to build decision tree for")
@click.option("--top-k", default=5, show_default=True, help="Top-K candidates")
def decision_tree_cmd(repo_path: str, prompt: str, top_k: int) -> None:
    """Build and display the temporary decision tree for a prompt or bug description."""
    from pathlib import Path
    from graph_agent.db.connection import get_session
    from graph_agent.db.models import Repository
    from graph_agent.indexer.ingester import index_repository
    from graph_agent.graph.builder import build_graph
    from graph_agent.graph.decision_tree import build_temporary_decision_tree

    root = str(Path(repo_path).resolve())
    with get_session() as session:
        repo = session.query(Repository).filter_by(root_path=root).first()
        if repo is None:
            console.print(f"[cyan]Indexing {root}...[/cyan]")
            index_repository(session, root)
            repo = session.query(Repository).filter_by(root_path=root).first()
            build_graph(session, repo.id)

        dt, context, _ = build_temporary_decision_tree(session, repo.id, prompt, top_k=top_k)
        console.print(dt.to_rich_tree())


# ---------------------------------------------------------------------------
# graph-agent localize
# ---------------------------------------------------------------------------
@cli.command("localize")
@click.argument("repo_path", type=click.Path(exists=True, file_okay=False))
@click.option("--description", default="", help="Bug description")
@click.option("--stack-trace", default="", help="Stack trace text")
@click.option("--failing-test", default="", help="Failing test name/path")
@click.option("--top-k", default=10, show_default=True, help="Top-K candidates to display")
def localize_cmd(repo_path: str, description: str, stack_trace: str,
                 failing_test: str, top_k: int) -> None:
    """Rank suspicious functions using the code graph and bug evidence."""
    from pathlib import Path
    from graph_agent.db.connection import get_session
    from graph_agent.db.models import Repository
    from graph_agent.localizer.ranker import localize

    with get_session() as session:
        root = str(Path(repo_path).resolve())
        repo = session.query(Repository).filter_by(root_path=root).first()
        if repo is None:
            console.print("[red]Repo not indexed. Run `graph-agent index` first.[/red]")
            import sys; sys.exit(1)

        candidates = localize(
            session, repo.id,
            description=description,
            stack_trace=stack_trace,
            failing_test=failing_test,
            top_k=top_k,
        )

    if not candidates:
        console.print("[yellow]No candidates found — provide more evidence.[/yellow]")
        return

    t = Table(title=f"Top-{top_k} Suspicious Symbols", show_header=True)
    t.add_column("Rank", justify="right")
    t.add_column("Symbol", style="bold")
    t.add_column("Score", justify="right")
    t.add_column("Evidence", style="dim")
    t.add_column("Source", style="dim")
    for i, c in enumerate(candidates):
        t.add_row(
            str(i + 1),
            c.qualified_name,
            f"{c.score:.2f}",
            c.evidence_kind.value,
            c.evidence_source,
        )
    console.print(t)


# ---------------------------------------------------------------------------
# graph-agent repair
# ---------------------------------------------------------------------------
@cli.command("repair")
@click.argument("repo_path", type=click.Path(exists=True, file_okay=False))
@click.option("--description", default="", help="Bug description")
@click.option("--stack-trace", default="", help="Stack trace text")
@click.option("--failing-test", default="", help="Failing test name/path")
@click.option("--adapter", default=None, envvar="GRAPH_AGENT_LLM_ADAPTER",
              help="LLM adapter: mock | openai_compat")
@click.option("--output-diff", default=None, type=click.Path(),
              help="Write generated diff to this file")
def repair_cmd(repo_path: str, description: str, stack_trace: str,
               failing_test: str, adapter: str | None, output_diff: str | None) -> None:
    """Run the full repair pipeline: localize → context → LLM → verify."""
    import sys
    from pathlib import Path
    from graph_agent.db.connection import get_session
    from graph_agent.db.models import Repository
    from graph_agent.graph.builder import build_graph
    from graph_agent.localizer.ranker import localize
    from graph_agent.context.selector import BudgetContextSelector
    from graph_agent.llm.openai_compat import get_adapter
    from graph_agent.patch.generator import generate_patch
    from graph_agent.patch.validator import validate_patch
    from graph_agent.settings import get_settings

    s = get_settings()

    with get_session() as session:
        root = str(Path(repo_path).resolve())
        repo = session.query(Repository).filter_by(root_path=root).first()
        if repo is None:
            console.print("[red]Repo not indexed. Run `graph-agent index` first.[/red]")
            sys.exit(1)

        # Ensure graph is built
        build_graph(session, repo.id)

        candidates = localize(
            session, repo.id,
            description=description,
            stack_trace=stack_trace,
            failing_test=failing_test,
        )
        if not candidates:
            console.print("[yellow]No localization candidates. Aborting repair.[/yellow]")
            sys.exit(1)

        ctx = BudgetContextSelector().select(
            session, repo.id, candidates, s.context.token_budget
        )

    llm = get_adapter(adapter)
    console.print(f"[cyan]LLM adapter:[/cyan] {llm.name()}")
    console.print(f"[cyan]Candidates:[/cyan] {len(candidates)}, "
                  f"Context slices: {len(ctx.slices)} ({ctx.total_tokens_estimate} tokens)")

    patch = generate_patch(
        adapter=llm,
        description=description,
        stack_trace=stack_trace,
        failing_test=failing_test,
        candidates=candidates,
        context=ctx,
    )

    if not patch.is_valid:
        console.print("[red]LLM did not return a valid diff.[/red]")
        sys.exit(1)

    console.print("\n[bold green]Root Cause[/bold green]")
    console.print(patch.root_cause.strip())

    val = validate_patch(repo_path, patch.diff)
    status = "[green][OK] Applies cleanly[/green]" if val.applies_cleanly else f"[red][ERROR] {val.error}[/red]"
    console.print(f"\n[bold]Patch validation:[/bold] {status}")

    if output_diff:
        Path(output_diff).write_text(patch.diff, encoding="utf-8")
        console.print(f"\nDiff written to [cyan]{output_diff}[/cyan]")
    else:
        console.print("\n[bold]Generated Diff[/bold]")
        console.print(patch.diff)

    console.print(
        "\n[yellow][WARN] Patch NOT applied.[/yellow] "
        "Review the diff above and apply manually with:\n"
        f"  git apply <diff_file>"
    )


# ---------------------------------------------------------------------------
# graph-agent models
# ---------------------------------------------------------------------------
@cli.command("models")
@click.option("--api-base", default=None, envvar="OLLAMA_API_BASE",
              help="Ollama API base URL (defaults to http://localhost:11434/v1)")
def models_cmd(api_base: str | None) -> None:
    """List local models available in the local Ollama instance."""
    from graph_agent.llm.ollama import list_ollama_models, get_ollama_host_url
    host = get_ollama_host_url(api_base)
    models = list_ollama_models(host)
    console.print(f"[bold cyan]Local Ollama instance:[/bold cyan] {host}")
    if not models:
        console.print("[yellow]No local models found or Ollama server is not running.[/yellow]")
        console.print("[dim]Run 'ollama serve' and 'ollama pull gptoss:20b' to get started.[/dim]")
    else:
        t = Table(title="Installed Ollama Models", show_header=True)
        t.add_column("#", justify="right")
        t.add_column("Model Name", style="bold green")
        for i, m in enumerate(models, 1):
            t.add_row(str(i), m)
        console.print(t)


# ---------------------------------------------------------------------------
# graph-agent chat
# ---------------------------------------------------------------------------
@cli.command("chat")
@click.argument("repo_path", type=click.Path(exists=True, file_okay=False))
@click.option("--adapter", default=None, envvar="GRAPH_AGENT_LLM_ADAPTER",
              help="LLM adapter: ollama | openai_compat | mock")
@click.option("--model", default=None, envvar="OLLAMA_MODEL",
              help="Ollama model name (e.g. gptoss:20b, llama3). If omitted, an interactive dropdown is shown.")
@click.option("--api-base", default=None, envvar="OLLAMA_API_BASE",
              help="Ollama API base URL (e.g. http://localhost:11434/v1)")
@click.option("--api-key", default=None, envvar="OPENAI_API_KEY",
              help="Optional API key (defaults to 'ollama')")
def chat_cmd(repo_path: str, adapter: str | None, model: str | None, api_base: str | None, api_key: str | None) -> None:
    """Start an interactive chat session with real-time prompt rendering and local Ollama models."""
    from graph_agent.chat.session import run_chat_session
    run_chat_session(
        repo_path,
        adapter_name=adapter,
        model_name=model,
        api_base=api_base,
        api_key=api_key,
    )


if __name__ == "__main__":
    cli()

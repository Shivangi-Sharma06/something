"""Temporary Decision Tree module — constructs and visualizes per-prompt decision reasoning.

When a user submits a prompt, this module maps the prompt to the repository AST graph,
extracts evidence, traces caller/callee neighborhoods, and forms a Temporary Decision Tree:
  Root: User Goal / Prompt Intent
  ├── Files Identified
  ├── Target Symbols (with score, evidence source, and caller/callee context)
  ├── Context Decisions (full source vs signature, lines selected)
  └── Planned Actions (Q&A analysis or patch formulation)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from rich.tree import Tree
from sqlalchemy.orm import Session

from graph_agent.db.models import Repository, File, Symbol
from graph_agent.localizer.ranker import localize, ScoredSymbol
from graph_agent.localizer.evidence import extract_evidence
from graph_agent.graph.queries import get_callers, get_callees
from graph_agent.context.selector import BudgetContextSelector, SelectedContext
from graph_agent.settings import get_settings

log = logging.getLogger(__name__)


@dataclass
class DecisionSymbolNode:
    symbol_id: int
    qualified_name: str
    kind: str
    rel_path: str
    line_start: int | None
    line_end: int | None
    score: float
    reason: str
    evidence_kind: str
    evidence_source: str
    is_full_context: bool = False
    callers: list[str] = field(default_factory=list)
    callees: list[str] = field(default_factory=list)


@dataclass
class TemporaryDecisionTree:
    prompt: str
    repo_id: int
    repo_root: str
    intent: str  # "repair" | "qa_explanation" | "refactor"
    target_files: set[str] = field(default_factory=set)
    symbol_nodes: list[DecisionSymbolNode] = field(default_factory=list)
    token_budget: int = 8000
    tokens_used: int = 0
    context_slices_count: int = 0
    summary_rationale: str = ""

    def to_rich_tree(self) -> Tree:
        """Convert the decision tree to a rich Tree for visual display."""
        root = Tree(
            f"[bold magenta]Temporary Decision Tree[/bold magenta] "
            f"[dim](Intent: [cyan]{self.intent}[/cyan] | Targets: {len(self.symbol_nodes)} symbols in {len(self.target_files)} files)[/dim]"
        )

        intent_node = root.add(f"[bold yellow]Objective:[/bold yellow] [italic]{self.prompt}[/italic]")

        # Files & Symbols branch
        files_branch = root.add(f"[bold blue]Repository Targets ({len(self.target_files)} file(s))[/bold blue]")
        for fpath in sorted(self.target_files):
            file_tree = files_branch.add(f"[bold cyan]{fpath}[/bold cyan]")
            file_syms = [s for s in self.symbol_nodes if s.rel_path == fpath]
            for s in file_syms:
                lines = f"L{s.line_start}-L{s.line_end}" if s.line_start else "file scope"
                ctx_type = "[green][Full Source][/green]" if s.is_full_context else "[yellow][Signature][/yellow]"
                sym_tree = file_tree.add(
                    f"[bold]{s.qualified_name}[/bold] ({s.kind}, {lines}) "
                    f"Score: [bold green]{s.score:.2f}[/bold green] {ctx_type}"
                )
                sym_tree.add(f"[dim]Rationale: {s.reason}[/dim]")
                if s.callers:
                    sym_tree.add(f"[dim]<- Callers: {', '.join(s.callers[:3])}[/dim]")
                if s.callees:
                    sym_tree.add(f"[dim]-> Callees: {', '.join(s.callees[:3])}[/dim]")

        # Context Budget branch
        ctx_branch = root.add(
            f"[bold green]Context Budget Decision[/bold green]: "
            f"{self.tokens_used}/{self.token_budget} tokens allocated across {self.context_slices_count} slice(s)"
        )

        # Planned Action branch
        action_desc = (
            "Synthesize minimal unified patch diff and root cause"
            if self.intent in ("repair", "refactor")
            else "Provide detailed answer with code references and structural analysis"
        )
        root.add(f"[bold yellow]Action Plan:[/bold yellow] {action_desc}")

        return root

    def to_prompt_context(self) -> str:
        """Returns structured decision tree reasoning to inject into LLM prompt."""
        lines = [
            "### Decision Tree Reasoning",
            f"- Goal / Intent: {self.intent}",
            f"- Target Files: {', '.join(self.target_files) if self.target_files else 'None'}",
            "- Selected Symbols & Rationale:",
        ]
        for s in self.symbol_nodes:
            rel = f" in {s.rel_path}"
            callers_info = f" (called by: {', '.join(s.callers)})" if s.callers else ""
            callees_info = f" (calls: {', '.join(s.callees)})" if s.callees else ""
            lines.append(f"  * `{s.qualified_name}`{rel}: {s.reason}{callers_info}{callees_info}")
        return "\n".join(lines)


def classify_prompt_intent(prompt: str) -> str:
    """Classifies user intent into repair/fix, refactor, or qa_explanation."""
    p = prompt.lower()
    if any(k in p for k in ("fix", "bug", "error", "fail", "exception", "traceback", "patch", "repair", "issue")):
        return "repair"
    elif any(k in p for k in ("refactor", "rename", "add function", "add method", "implement", "update code", "modify")):
        return "refactor"
    else:
        return "qa_explanation"


def build_temporary_decision_tree(
    session: Session,
    repo_id: int,
    user_prompt: str,
    top_k: int = 5,
) -> tuple[TemporaryDecisionTree, SelectedContext, list[ScoredSymbol]]:
    """Builds a temporary decision tree and extracts selected repository context."""
    s = get_settings()
    repo = session.query(Repository).filter_by(id=repo_id).first()
    repo_root = repo.root_path if repo else ""

    intent = classify_prompt_intent(user_prompt)

    # Localize candidates using bug evidence / keywords / stack trace
    candidates: list[ScoredSymbol] = localize(
        session,
        repo_id,
        description=user_prompt,
        stack_trace=user_prompt if ("Traceback" in user_prompt or "line " in user_prompt) else "",
        failing_test=user_prompt if "test_" in user_prompt else "",
        top_k=top_k,
    )

    # If prompt mentions specific symbols or files, make sure they are in candidates
    token_budget = s.context.token_budget
    context = BudgetContextSelector().select(session, repo_id, candidates, token_budget)

    # Map which slices got full source
    full_source_sym_names = {slc.qualified_name for slc in context.slices if slc.is_full}

    target_files: set[str] = set()
    symbol_nodes: list[DecisionSymbolNode] = []

    for cand in candidates:
        sym = session.query(Symbol).filter_by(id=cand.symbol_id).first()
        if not sym:
            continue

        file_obj = session.query(File).filter_by(id=sym.file_id).first()
        rel_path = file_obj.rel_path if file_obj else "unknown"
        target_files.add(rel_path)

        callers_rows = get_callers(session, repo_id, sym.id, max_depth=1)
        callees_rows = get_callees(session, repo_id, sym.id, max_depth=1)

        callers = [c["qualified_name"].split(".")[-1] for c in callers_rows]
        callees = [c["qualified_name"].split(".")[-1] for c in callees_rows]

        reason = (
            f"Matched via {cand.evidence_source} ({cand.evidence_kind.value}) "
            f"with relevance score {cand.score:.2f}."
        )

        symbol_nodes.append(DecisionSymbolNode(
            symbol_id=sym.id,
            qualified_name=cand.qualified_name,
            kind=sym.kind,
            rel_path=rel_path,
            line_start=sym.line_start,
            line_end=sym.line_end,
            score=cand.score,
            reason=reason,
            evidence_kind=cand.evidence_kind.value,
            evidence_source=cand.evidence_source,
            is_full_context=cand.qualified_name in full_source_sym_names,
            callers=callers,
            callees=callees,
        ))

    summary_rationale = (
        f"Selected {len(target_files)} target file(s) and {len(symbol_nodes)} symbol(s) "
        f"for {intent} based on structural graph traversal."
    )

    decision_tree = TemporaryDecisionTree(
        prompt=user_prompt,
        repo_id=repo_id,
        repo_root=repo_root,
        intent=intent,
        target_files=target_files,
        symbol_nodes=symbol_nodes,
        token_budget=token_budget,
        tokens_used=context.total_tokens_estimate,
        context_slices_count=len(context.slices),
        summary_rationale=summary_rationale,
    )

    return decision_tree, context, candidates

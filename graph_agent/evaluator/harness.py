"""Evaluation harness — compare retrieval strategies on the same tasks.

Strategies:
  no_retrieval    — send bug report to LLM with no code context
  graph_retrieval — full pipeline: index → graph → localize → context → LLM

Output: CSV + JSON in eval_results/ directory.

ADR-7: Vector retrieval is stubbed (not yet implemented).
"""

from __future__ import annotations

import csv
import json
import logging
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

from graph_agent.settings import get_settings

log = logging.getLogger(__name__)


@dataclass
class EvalTask:
    task_id: str
    repo_root: str
    description: str
    stack_trace: str
    failing_test: str
    buggy_function: str   # ground truth — the function that contains the bug


@dataclass
class EvalResult:
    task_id: str
    strategy: str
    success: bool
    localization_rank: int     # rank of buggy_function in candidates (-1 if not found)
    tokens_used: int
    retries: int
    latency_s: float
    test_pass_rate: float
    error: str = ""
    extra: dict[str, Any] = field(default_factory=dict)


def _run_no_retrieval(
    task: EvalTask, adapter, max_retries: int
) -> EvalResult:
    """Baseline: no code context at all."""
    from graph_agent.patch.generator import build_repair_prompt, extract_patch
    from graph_agent.patch.validator import validate_patch
    from graph_agent.verifier.runner import run_tests_in_isolation

    start = time.time()
    messages = build_repair_prompt(
        description=task.description,
        stack_trace=task.stack_trace,
        failing_test=task.failing_test,
        candidates=[],
        context=type("C", (), {"slices": []})(),  # empty context
    )
    # Count tokens roughly
    total_chars = sum(len(m["content"]) for m in messages)
    tokens = total_chars // 4

    patch = extract_patch(adapter.complete(messages))
    if not patch.is_valid:
        return EvalResult(
            task_id=task.task_id, strategy="no_retrieval",
            success=False, localization_rank=-1,
            tokens_used=tokens, retries=0,
            latency_s=time.time() - start,
            test_pass_rate=0.0, error="No valid diff",
        )

    val = validate_patch(task.repo_root, patch.diff)
    if not val.applies_cleanly:
        return EvalResult(
            task_id=task.task_id, strategy="no_retrieval",
            success=False, localization_rank=-1,
            tokens_used=tokens, retries=0,
            latency_s=time.time() - start,
            test_pass_rate=0.0, error=val.error,
        )

    vresult = run_tests_in_isolation(task.repo_root, patch_diff=patch.diff)
    total = vresult.tests_passed + vresult.tests_failed
    rate = vresult.tests_passed / total if total else 0.0

    return EvalResult(
        task_id=task.task_id, strategy="no_retrieval",
        success=vresult.success, localization_rank=-1,
        tokens_used=tokens, retries=0,
        latency_s=time.time() - start,
        test_pass_rate=rate,
    )


def _run_graph_retrieval(
    task: EvalTask, adapter, session, repo_id: int, max_retries: int
) -> EvalResult:
    """Full graph-based pipeline."""
    from graph_agent.localizer.ranker import localize
    from graph_agent.context.selector import BudgetContextSelector
    from graph_agent.patch.generator import generate_patch
    from graph_agent.patch.validator import validate_patch
    from graph_agent.verifier.runner import run_tests_in_isolation
    from graph_agent.settings import get_settings

    start = time.time()
    s = get_settings()

    candidates = localize(
        session, repo_id,
        description=task.description,
        stack_trace=task.stack_trace,
        failing_test=task.failing_test,
    )

    # Find rank of ground truth
    rank = -1
    for i, c in enumerate(candidates):
        if task.buggy_function and task.buggy_function in c.qualified_name:
            rank = i + 1
            break

    selector = BudgetContextSelector()
    context = selector.select(session, repo_id, candidates, s.context.token_budget)

    patch = generate_patch(
        adapter=adapter,
        description=task.description,
        stack_trace=task.stack_trace,
        failing_test=task.failing_test,
        candidates=candidates,
        context=context,
    )

    tokens = context.total_tokens_estimate
    if not patch.is_valid:
        return EvalResult(
            task_id=task.task_id, strategy="graph_retrieval",
            success=False, localization_rank=rank,
            tokens_used=tokens, retries=0,
            latency_s=time.time() - start,
            test_pass_rate=0.0, error="No valid diff",
        )

    val = validate_patch(task.repo_root, patch.diff)
    if not val.applies_cleanly:
        return EvalResult(
            task_id=task.task_id, strategy="graph_retrieval",
            success=False, localization_rank=rank,
            tokens_used=tokens, retries=0,
            latency_s=time.time() - start,
            test_pass_rate=0.0, error=val.error,
        )

    vresult = run_tests_in_isolation(task.repo_root, patch_diff=patch.diff)
    total = vresult.tests_passed + vresult.tests_failed
    rate = vresult.tests_passed / total if total else 0.0

    return EvalResult(
        task_id=task.task_id, strategy="graph_retrieval",
        success=vresult.success, localization_rank=rank,
        tokens_used=tokens, retries=0,
        latency_s=time.time() - start,
        test_pass_rate=rate,
    )


def run_evaluation(
    tasks: list[EvalTask],
    adapter,
    session=None,
    repo_id: int = 0,
    output_dir: str | Path | None = None,
    strategies: list[str] | None = None,
    max_retries: int = 1,
) -> list[EvalResult]:
    """Run all tasks under all configured strategies and write results."""
    s = get_settings()
    strategies = strategies or s.evaluator.strategies
    out_dir = Path(output_dir or s.evaluator.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_results: list[EvalResult] = []

    for task in tasks:
        for strategy in strategies:
            log.info("Evaluating task=%s strategy=%s", task.task_id, strategy)
            try:
                if strategy == "no_retrieval":
                    result = _run_no_retrieval(task, adapter, max_retries)
                elif strategy == "graph_retrieval" and session and repo_id:
                    result = _run_graph_retrieval(task, adapter, session, repo_id, max_retries)
                else:
                    log.warning("Skipping strategy %s (not configured)", strategy)
                    continue
            except Exception as exc:
                log.error("Task %s strategy %s failed: %s", task.task_id, strategy, exc)
                result = EvalResult(
                    task_id=task.task_id, strategy=strategy,
                    success=False, localization_rank=-1,
                    tokens_used=0, retries=0, latency_s=0.0,
                    test_pass_rate=0.0, error=str(exc),
                )
            all_results.append(result)

    # Write CSV
    csv_path = out_dir / "eval_results.csv"
    json_path = out_dir / "eval_results.json"

    fieldnames = [
        "task_id", "strategy", "success", "localization_rank",
        "tokens_used", "retries", "latency_s", "test_pass_rate", "error",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for r in all_results:
            d = asdict(r)
            d.pop("extra", None)
            writer.writerow({k: d.get(k, "") for k in fieldnames})

    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump([asdict(r) for r in all_results], fh, indent=2)

    log.info("Evaluation complete. Results: %s", csv_path)
    return all_results

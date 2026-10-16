"""Patch generator — constructs LLM prompts and extracts unified diffs from responses."""

from __future__ import annotations

import re
import logging
from dataclasses import dataclass

from graph_agent.context.selector import SelectedContext
from graph_agent.localizer.ranker import ScoredSymbol
from graph_agent.llm.base import ModelAdapter

log = logging.getLogger(__name__)

SYSTEM_PROMPT = """\
You are an expert software engineer performing automated bug repair.
Given a bug report and relevant source code context, you will:
1. Identify the root cause of the bug with a brief explanation.
2. Produce a MINIMAL unified diff (git diff format) that fixes the bug.

Rules:
- Fix only what is necessary. Do not refactor unrelated code.
- The diff must be valid and apply cleanly with `git apply`.
- Explain the root cause FIRST, then provide the diff.
- Wrap the diff in a ```diff ... ``` code block.
- Do not include any other code blocks.
"""


def _format_context(context: SelectedContext) -> str:
    """Format selected context slices into a readable string for the prompt."""
    parts = []
    for slc in context.slices:
        label = "full source" if slc.is_full else "signature only"
        parts.append(
            f"### {slc.qualified_name} ({label})\n"
            f"# File: {slc.rel_path} lines {slc.line_start}-{slc.line_end}\n"
            f"```python\n{slc.content}\n```"
        )
    return "\n\n".join(parts)


def _format_candidates(candidates: list[ScoredSymbol]) -> str:
    lines = []
    for i, sym in enumerate(candidates[:5]):
        lines.append(
            f"{i+1}. `{sym.qualified_name}` (score={sym.score:.2f}, "
            f"evidence={sym.evidence_kind.value}, source={sym.evidence_source})"
        )
    return "\n".join(lines)


def build_repair_prompt(
    description: str,
    stack_trace: str,
    failing_test: str,
    candidates: list[ScoredSymbol],
    context: SelectedContext,
) -> list[dict]:
    """Build the OpenAI-style messages list for the repair request."""
    user_content = f"""\
## Bug Report

**Description:** {description or "(none)"}

**Stack Trace:**
```
{stack_trace or "(none)"}
```

**Failing Test:** `{failing_test or "(none)"}`

## Suspicious Functions (ranked)
{_format_candidates(candidates)}

## Relevant Source Code
{_format_context(context)}

## Task
Identify the root cause and produce a minimal unified diff to fix the bug.
"""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]


@dataclass
class GeneratedPatch:
    raw_response: str
    root_cause: str
    diff: str          # raw unified diff text
    is_valid: bool     # True if a diff block was successfully extracted


def extract_patch(response: str) -> GeneratedPatch:
    """Parse the LLM response to extract root cause and diff."""
    # Extract diff from ```diff ... ``` block
    diff_match = re.search(r'```diff\s*\n(.*?)```', response, re.DOTALL)
    diff = diff_match.group(1).strip() if diff_match else ""

    # Extract root cause: everything before the diff block (or first 500 chars)
    root_cause = response.split("```diff")[0].strip() if "```diff" in response else response[:500]

    return GeneratedPatch(
        raw_response=response,
        root_cause=root_cause,
        diff=diff,
        is_valid=bool(diff),
    )


def generate_patch(
    adapter: ModelAdapter,
    description: str,
    stack_trace: str,
    failing_test: str,
    candidates: list[ScoredSymbol],
    context: SelectedContext,
) -> GeneratedPatch:
    """Call the LLM and return the parsed patch."""
    messages = build_repair_prompt(
        description=description,
        stack_trace=stack_trace,
        failing_test=failing_test,
        candidates=candidates,
        context=context,
    )
    log.info("Calling %s for patch generation ...", adapter.name())
    response = adapter.complete(messages)
    patch = extract_patch(response)
    if not patch.is_valid:
        log.warning("LLM response did not contain a valid diff block")
    return patch

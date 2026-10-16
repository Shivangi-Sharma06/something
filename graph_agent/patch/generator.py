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
You are an expert software engineer performing automated code modifications and bug repair.
Given the user prompt and relevant source code context, your output MUST follow this exact structure:

### Summary of Changes & Reasoning
Provide a concise 2 to 4 sentence explanation:
- What change you are making.
- Why this change solves the user prompt based on the code context.

### Unified Patch
Produce a minimal unified diff inside a ```diff code block.

Strict Diff Rules:
- Header:
  --- a/<filepath>
  +++ b/<filepath>
- Context lines must exactly match the source code provided in the prompt.
- Do NOT output stream-of-consciousness thought traces. Keep the summary clear, professional, and brief.
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
    # Collect full file contents for candidates so the model sees complete real code
    file_sections = []
    seen_files = set()
    for slc in context.slices:
        if slc.rel_path not in seen_files:
            seen_files.add(slc.rel_path)
            file_sections.append(
                f"### File: {slc.rel_path}\n"
                f"```python\n{slc.content}\n```"
            )

    user_content = f"""\
## User Request
{description}

{f"**Stack Trace:**\\n```\\n{stack_trace}\\n```" if stack_trace else ""}
{f"**Failing Test:** `{failing_test}`" if failing_test else ""}

## Code Context
{_format_context(context)}

## Instructions
1. Provide a clear "Reasoning:" paragraph explaining:
   - Why this change is needed.
   - What logic is modified or added.
2. Provide a unified diff inside ```diff ... ```:
   - Header: `--- a/<filepath>` and `+++ b/<filepath>`
   - Use the EXACT code from the Code Context above for unchanged context lines.
   - Do NOT invent or assume different class structures.
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
    # 1. Try ```diff ... ``` or ```patch ... ```
    diff_match = re.search(r'```(?:diff|patch)\s*\n(.*?)```', response, re.DOTALL)
    
    # 2. Fallback: try any ``` block that contains --- a/ or @@
    if not diff_match:
        for block in re.finditer(r'```[a-zA-Z0-9_-]*\s*\n(.*?)```', response, re.DOTALL):
            candidate = block.group(1).strip()
            if "--- " in candidate or "@@" in candidate:
                diff_match = block
                break

    diff = diff_match.group(1).strip() if diff_match else ""

    # Extract root cause: text before the diff block
    if diff_match:
        root_cause = response[: diff_match.start()].strip()
    else:
        root_cause = response.strip()

    # Clean up empty root cause
    if not root_cause:
        root_cause = "Code modification requested per prompt context."

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

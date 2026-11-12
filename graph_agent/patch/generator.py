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
You are an autonomous software engineering agent tasked with modifying code in a repository.
IMPORTANT: Never tell the user how to make the change themselves. Do NOT write "You can change this by..." or "Here is how you can do it".
Instead, YOU must make the decisions, determine the exact code changes needed, and produce the complete unified patch so the agent can apply it directly to the repository files upon user approval.

Your response MUST follow this exact structure:

### Decisions and Considered Factors
- What target files, functions, and call dependencies were considered.
- Why this approach was selected based on the code context.

### Changes Made
- Concise, concrete bullet points detailing all modifications, additions, or removals.

### Unified Patch
Produce the minimal unified diff inside a ```diff code block ready to apply:
```diff
--- a/<filepath>
+++ b/<filepath>
@@ ... @@
...
```

Strict Diff Rules:
- Header: --- a/<filepath> and +++ b/<filepath>
- Context lines must match the provided repository code.
- Provide real, complete code without placeholders.
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


from typing import Callable, Optional

SYSTEM_QA_PROMPT = """\
You are an expert software engineering agent analyzing a software repository.
Answer the user's question accurately using the provided code context and decision tree reasoning.
If the user asks for any code modification or implementation:
- Do NOT advise the user how they can do it.
- State your Decisions and Considered Factors.
- State the Changes Made.
- Provide the complete unified diff inside ```diff ... ``` so it can be applied directly to the codebase with user confirmation.
"""


def build_repair_prompt(
    description: str,
    stack_trace: str = "",
    failing_test: str = "",
    candidates: list[ScoredSymbol] | None = None,
    context: SelectedContext | None = None,
    decision_tree_reasoning: str | None = None,
) -> list[dict]:
    # Collect full file contents for candidates so the model sees complete real code
    file_sections = []
    seen_files = set()
    if context:
        for slc in context.slices:
            if slc.rel_path not in seen_files:
                seen_files.add(slc.rel_path)
                file_sections.append(
                    f"### File: {slc.rel_path}\n"
                    f"```python\n{slc.content}\n```"
                )

    code_ctx = _format_context(context) if context else ""
    dt_section = f"\n\n{decision_tree_reasoning}" if decision_tree_reasoning else ""

    user_content = f"""\
## User Request
{description}

{f"**Stack Trace:**\\n```\\n{stack_trace}\\n```" if stack_trace else ""}
{f"**Failing Test:** `{failing_test}`" if failing_test else ""}
{dt_section}

## Code Context
{code_ctx}

## Instructions
1. State your Decisions and Considered Factors:
   - What target files, functions, and architecture factors were evaluated.
2. State the Changes Made:
   - Specific list of code modifications implemented.
3. Provide the Unified Patch:
   - Output the exact unified diff inside a ```diff code block ready for application.
   - Do NOT give advisory steps telling the user how to do it; provide the actual code diff.
"""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]


def build_qa_prompt(
    description: str,
    context: SelectedContext | None = None,
    decision_tree_reasoning: str | None = None,
) -> list[dict]:
    code_ctx = _format_context(context) if context else ""
    dt_section = f"\n\n{decision_tree_reasoning}" if decision_tree_reasoning else ""

    user_content = f"""\
## Question / Inquiry
{description}
{dt_section}

## Code Context from Repository
{code_ctx}

Please provide a detailed, accurate response based on the repository source code and structural relationships above.
"""
    return [
        {"role": "system", "content": SYSTEM_QA_PROMPT},
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
    stack_trace: str = "",
    failing_test: str = "",
    candidates: list[ScoredSymbol] | None = None,
    context: SelectedContext | None = None,
    decision_tree_reasoning: str | None = None,
    stream_chunk_cb: Callable[[str], None] | None = None,
) -> GeneratedPatch:
    """Call the LLM and return the parsed patch, optionally streaming chunks in real time."""
    messages = build_repair_prompt(
        description=description,
        stack_trace=stack_trace,
        failing_test=failing_test,
        candidates=candidates or [],
        context=context or SelectedContext(),
        decision_tree_reasoning=decision_tree_reasoning,
    )
    log.info("Calling %s for patch generation ...", adapter.name())
    if stream_chunk_cb is not None:
        chunks = []
        for chunk in adapter.stream(messages):
            stream_chunk_cb(chunk)
            chunks.append(chunk)
        response = "".join(chunks)
    else:
        response = adapter.complete(messages)

    patch = extract_patch(response)
    if not patch.is_valid:
        log.warning("LLM response did not contain a valid diff block")
    return patch


def generate_answer(
    adapter: ModelAdapter,
    description: str,
    context: SelectedContext | None = None,
    decision_tree_reasoning: str | None = None,
    stream_chunk_cb: Callable[[str], None] | None = None,
) -> str:
    """Call the LLM for Q&A / repository understanding with real-time streaming."""
    messages = build_qa_prompt(
        description=description,
        context=context or SelectedContext(),
        decision_tree_reasoning=decision_tree_reasoning,
    )
    log.info("Calling %s for Q&A ...", adapter.name())
    if stream_chunk_cb is not None:
        chunks = []
        for chunk in adapter.stream(messages):
            stream_chunk_cb(chunk)
            chunks.append(chunk)
        return "".join(chunks)
    else:
        return adapter.complete(messages)

# LIMITATIONS.md

> **Scope**: This document records known limitations, design trade-offs, and
> deferred capabilities of the Graph-Based Repository-Level Coding Agent as
> implemented in the first milestone.

---

## 1. Language Support

| Status | Item |
|--------|------|
| ✅ Implemented | Python (via `tree-sitter-python`) |
| 🔲 Planned | JavaScript / TypeScript (`tree-sitter-javascript`, `tree-sitter-typescript`) |
| 🔲 Planned | Java, Go, Rust — grammar bindings available but not wired |

The walker and ingester are language-agnostic; adding a new language requires
only a grammar wheel install and a one-line entry in `config/default.toml`
`[indexer.extensions]`.

---

## 2. Edge Types

Currently built:
- `CONTAINS` — parent symbol encloses child symbol (class→method, etc.)
- `CALLS` — function/method calls another function/method

Not yet built:
- `INHERITS` — class inherits from another class (requires type inference)
- `HANDLES` — exception handler relationships
- `HAS_TYPE` — parameter/return type annotations (requires type stub resolution)
- `TESTS` — test function → function-under-test linkage

---

## 3. Call Resolution Confidence

The heuristic name-match resolver (`confidence=0.7`) does **not** perform
type inference. For overloaded methods or common names (e.g. `get`, `run`),
it picks the same-file symbol first, then any repo symbol alphabetically.
This can produce false positives.

A future improvement: integrate `jedi` or `pyright` for static name resolution
to boost `confidence` to `1.0` on verified matches.

---

## 4. pgvector / Vector Retrieval

`pgvector` is not installed on the PostgreSQL 18.6 instance in this environment.
The `vector_retrieval` comparison strategy in the evaluator is therefore stubbed
and returns empty results. Install `pgvector` and set `GRAPH_AGENT_EVAL_STRATEGIES`
to enable it.

---

## 5. LLM Adapter — Production Readiness

The `mock` adapter is used by default for all CI runs. For real repairs:
1. Set `GRAPH_AGENT_LLM_ADAPTER=openai_compat` in `.env`
2. Set `OPENAI_API_KEY` and optionally `OPENAI_API_BASE` (for Ollama/vLLM)
3. Set `OPENAI_MODEL` to the desired model (e.g. `gpt-4o`, `llama-3.1-70b`)

Retry back-off, rate-limit handling, and streaming are not yet implemented.

---

## 6. Verifier Isolation

The default isolation mode is `worktree` (temp directory copy). Docker-based
isolation (`GRAPH_AGENT_VERIFIER_ISOLATION=docker`) is declared in config but
the runner does not yet call Docker. A Docker implementation would provide
stronger reproducibility guarantees for multi-language repos.

---

## 7. Patch Auto-Application

**Patches are never auto-committed.** The `apply_patch()` function in
`graph_agent/patch/validator.py` exists but is not called by any automatic
pipeline step. Human approval is always required before applying to the
working tree. This is an intentional safety constraint.

---

## 8. Historical Co-Change Heuristic

The `file_co_change` weight slot in `LocalizationWeights` is wired into scoring
but the signal itself is not implemented (always 0.0). A future implementation
could read `git log --follow -p` to compute co-change frequency between files.

---

## 9. Evaluation Harness — Task Coverage

The evaluator ships with a single hardcoded task (the seeded `ZeroDivisionError`
fixture). For meaningful benchmarking, integrate SWE-bench or a project-local
bug dataset and pass tasks via `eval_tasks.json`.

---

## 10. Scalability

| Dimension | Current behaviour | Recommendation |
|-----------|-------------------|----------------|
| Repo size | Tested on 3-file fixture | Works up to ~10k file repos (no streaming) |
| Symbol count | ~14 symbols | PostgreSQL B-tree indexes support millions |
| CTE depth | Bounded to `max_traversal_depth=5` | Increase carefully; unbounded CTEs can lock |
| Session pooling | `pool_size=5` | Tune to connection count for multi-worker use |
| Parallel indexing | Single-threaded | Add async worker pool for large monorepos |

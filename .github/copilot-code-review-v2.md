### Summary

- **Scope**: Reviewed config, CLI, schema, Supabase client, services (ingestion, matching, categorization, embeddings), connector, Makefile, tests setup.
- **Goal**: Assess adherence to engineering contract (`AGENTS.md`) and surface gaps + prioritised fixes.

### Strengths

- **Config**: Uses `pydantic_settings` correctly with validators and modern union syntax.
- **Models**: Pydantic v2 usage, consistent `ConfigDict`, JSON parsing helpers.
- **Embeddings & external HTTP** (WordPress + OpenAI batch submit + embeddings): Proper `tenacity` retry on high-cost calls.
- **Schema**: Centralised in `src/data/schema.sql`, includes vector extension + helper RPC functions and view for export.
- **CLI**: Rich, modular commands covering full workflow, generally typed, clear user messaging.
- **Tests**: Broad unit test coverage across major components; fixtures isolate external calls with mocks.
- **Quality Gate**: Makefile target aggregates format, lint, types, coverage with fail-under 80.

### Key Gaps

- **Missing retry wrappers**: Supabase I/O (all CRUD and RPC calls), OpenAI batch status/retrieval/file content, httpx SQL RPC in init-db.
- **Logging & error handling**: Some silent fallbacks (e.g. embedding persistence warnings); lacks structured correlation IDs for workflow runs.
- **Import ordering**: Several modules (e.g., `src/cli.py`) not strictly alphabetical within stdlib/third-party blocks.
- **Singleton mismatch**: `get_settings()` docstring claims singleton but returns new instance each call; CLI mutates settings object.
- **Coverage configuration**: Large swaths (`src/services/*`, `src/connectors/*`, `supabase_client.py`) omitted; prefer selective exclusions.
- **Makefile parallel pytest**: Uses `-n 4` but `pytest-xdist` not listed in dev dependencies (will fail).
- **Supabase data constraints**: `matching_results` has `UNIQUE(taxonomy_id)`, blocking multiple historical matches.
- **Potential data race**: Batch workflow mutates settings flags in-place; subsequent commands may inherit altered config.
- **Rubric acceptance**: Entity overlap enforcement heuristic is opaque; lacks taxonomy context for keywords check.
- **Security / secrets**: `wordpress_vip_auth_token` presence not masked in logs.
- **File temp handling**: Batch JSONL written to tmp without cleanup/uniqueness safeguards; no size guard.
- **Ingestion embedding**: Serial embedding generation; should batch via `embed_batch`.
- **Test fixtures**: Mutate `os.environ` without cleanup fixture.
- **Data export**: `export_results` view uses empty string for missing target; ambiguous downstream.

### Risks

- Operational instability under transient Supabase or network blips due to missing retries.
- Silent logical drift (services omitted from coverage).
- Limited observability of matching evolution (single row constraint).
- Potential intermittent failure of quality-check pipeline (`-n 4` without xdist).
- Hidden state mutation via shared settings object in long-lived CLI processes.

### Recommended Actions (Priority Ordered)

- **High**: Add `tenacity` retry decorators to all `SupabaseClient` methods, OpenAI batch status/result retrieval, and httpx SQL RPC in init-db.
- **High**: Add `pytest-xdist` to `[project.optional-dependencies].dev` or remove `-n 4`.
- **High**: Revisit coverage omit list; include at least `src/services/*` and `src/connectors/*` with mocks.
- **High**: Replace `UNIQUE(taxonomy_id)` with composite uniqueness or remove constraint; consider `is_current` flag.
- **Medium**: Memoize `get_settings()` (module-level cache) or make settings immutable.
- **Medium**: Introduce structured logging (add `run_key` / `request_id`) across workflow + client.
- **Medium**: Enforce import ordering (use Ruff autofix) to meet contract.
- **Medium**: Batch embeddings during ingestion/taxonomy load via `embed_batch` before bulk upsert.
- **Medium**: Add cleanup fixture for environment variables (or use `monkeypatch`).
- **Medium**: Enhance batch file handling: per-run directory under `./data/batch/`, cleanup TTL.
- **Medium**: Improve export view: include `has_match` boolean; keep `similarity_score` NULL when unmatched.
- **Low**: Refine `_accept_by_rubric`: pass taxonomy context for entity enforcement; simplify + document.
- **Low**: Mask tokens in logs (log as boolean presence only).
- **Low**: Consider basic metrics (e.g., Prometheus counters) for match success/fail.

### Quick Wins

- Add `pytest-xdist`, run `ruff check --fix`, wrap Supabase methods with retry, and adjust coverage omit list.
- Memoize `get_settings()` or update docstring to avoid misleading singleton claim.
- Replace `UNIQUE(taxonomy_id)` constraint before expanding multi-stage auditing.

### Suggested Next Steps

- Want me to draft the retry wrappers + singleton fix + xdist dependency patch now?
- After structural changes: run `make quality-check`, adjust tests to cover added retry logic.
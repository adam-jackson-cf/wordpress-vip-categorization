<!-- 5ac5f2b8-85f1-4aad-9efd-5c5c5536fb9e 4bfb3d9f-1fc5-4d9d-9a27-5b4f7e2d6dd4 -->
# Reliability & Observability Hardening Plan

## Overview

Address the cross-cutting risks surfaced in the Copilot reviews by hardening retries, cleaning up schema/data contracts, reducing hidden state, and tightening our developer tooling. This plan groups all recommended actions into coherent workstreams so we can raise CI stability, improve analyst insight into matches, and keep the codebase aligned with our AGENTS contract.

## Implementation Steps

### P1 Stabilization Focus

The following tasks are mandatory for correct end-to-end runs and new-environment readiness. Complete them before tackling other workstreams.

1. **Resilience Guardrails**

- Add tenacity retries + typed errors for Supabase CRUD/RPC, OpenAI embeddings, batch submit/status/download, DSPy selector/judge calls, and httpx SQL helper.
- Ensure per-taxonomy DSPy failures fall back to `needs_human_review` rather than aborting the loop.

2. **Developer Tooling Baseline**

- Add `pytest-xdist` to the dev extras (or remove `-n 4`) so `make quality-check` works everywhere.
- Tighten coverage settings to include services/connectors/Supabase modules; keep only unavoidable exclusions.

3. **Configuration Hygiene**

- Memoize `get_settings()` and make every call return a fresh copy so CLI overrides can’t leak; stop mutating shared settings objects.

4. **Matching Data Model Fixes**

- Redesign `matching_results` to keep canonical state + history (`is_current` flag or composite key) and store semantic vs. LLM rubric scores separately.
- Persist best semantic candidate/score even when below threshold so analysts see context.

5. **OpenAI Batch File Handling**

- Store batch JSONL artifacts under per-run directories, guard against collisions, add cleanup, and replace the fixed 60s poll with adaptive intervals/jitter.

Once these P1 items are complete and quality checks are green, continue with the broader reliability and observability enhancements below.

### 1. Resilience & Error Handling Enhancements

**Files:** `src/data/supabase_client.py`, `src/services/embeddings.py`, `src/services/categorization.py`, `src/cli.py`, `src/services/workflow.py`, `src/connectors/wordpress_vip.py`, `tests/**`, `docs/OPERATIONS.md`

- Wrap every Supabase CRUD/RPC method with `tenacity` retries (covering timeouts, rate limits, connection errors) and raise typed exceptions so callers can distinguish fatal vs. transient failures.
- Expand retry coverage for OpenAI usage: embeddings, batch submission/status/download, and DSPy judge paths. Add jitter and log retry metadata.
- In `CategorizationService._find_best_match_llm`, catch DSPy/provider exceptions per taxonomy, log context, and downgrade that row to `needs_human_review` instead of aborting the loop.
- Replace regex-only HTML sanitization in `wordpress_vip` connector with a BeautifulSoup/readability extractor to mitigate injection and produce cleaner prompts.
- Add structured logging fields (`taxonomy_id`, `content_id`, `run_id`) and ensure secrets/tokens are never logged—log boolean presence instead.
- Update/extend tests to assert retries are configured (use `tenacity.Retrying.statistics`) and to simulate Supabase/OpenAI faults.

### 2. Matching Data Model & Schema Corrections

**Files:** `src/data/schema.sql`, `src/models.py`, `src/data/supabase_client.py`, `src/services/matching.py`, `src/services/categorization.py`, `src/exporters/csv_exporter.py`, `tests/**`, `docs/SCHEMA.md`

- Redesign `matching_results` uniqueness to allow historical rows (e.g., composite `UNIQUE(taxonomy_id, match_stage)` + `is_current` flag) so semantic + LLM stages can coexist while still exposing a canonical current match.
- Add a dedicated `llm_confidence` (or `rubric_topic_score`) column and field; semantic similarity should stay separate from LLM rubric metrics.
- When semantic thresholds fail, persist the top candidate id + score for analyst review instead of zeroing both fields.
- Update the export view so unmatched rows emit NULL target URLs and include a `has_match` boolean, avoiding ambiguous empty strings/zeroes.
- Ensure rubric JSON is always stored for accepted/flagged rows and surface summary stats in exporters/evaluator.
- Add/adjust migrations + Supabase client helpers, plus unit tests verifying schema constraints and data preservation.

### 3. Provider Abstractions & Embedding Efficiency

**Files:** `src/services/embeddings.py`, `src/services/categorization.py`, `src/services/matching.py`, `src/services/ingestion.py`, `src/config.py`, `src/optimization/dspy_optimizer.py`, `tests/**`, `docs/ARCHITECTURE.md`

- Introduce `EmbeddingProvider` and `LLMProvider` interfaces; wire CLI/services to instantiate providers based on `Settings`, enabling OpenRouter/Azure swaps.
- Refactor ingestion + matching to use `embed_batch()` (or new bulk API) so we don’t issue per-row embedding calls; persist embeddings as soon as they’re generated.
- Ensure DSPy selector vs. judge responsibilities are explicit: selector chooses index, judge scores rubric. Document this separation and add tests mocking each path.
- Memoize `get_settings()` (module-level cache) and stop mutating shared settings instances; have CLI derive immutable override copies for command-specific toggles.

### 4. Workflow Observability & File Handling

**Files:** `src/cli.py`, `src/services/workflow.py`, `src/services/categorization.py`, `src/services/ingestion.py`, `docs/WORKFLOW.md`, `tests/integration/**`

- Enhance OpenAI batch handling: write JSONL inputs/outputs to per-run directories under `data/batch/<run-id>/`, enforce unique filenames, guard file sizes, and clean up after success/failure.
- Replace the fixed 60s polling loop with adaptive intervals + jitter and early exit when counts match, reducing unnecessary waits.
- Capture best-candidate metadata (content_id, score, rubric) in workflow run stats so we can correlate acceptance rates per stage.
- Add structured status output + CLI commands to inspect recent runs, matching the workflow-run persistence we already built.

### 5. Developer Tooling & Test Hygiene

**Files:** `pyproject.toml`, `Makefile`, `tests/conftest.py`, `src/cli.py`, `src/**`, `docs/CONTRIBUTING.md`

- Add `pytest-xdist>=3` to the `dev` optional dependency (or drop `-n 4`); ensure CI installs matching extras.
- Tighten coverage configuration: remove broad exclusions for `src/services/*`, `src/connectors/*`, and `src/data/supabase_client.py`, keeping only files that truly require integration tests. Backfill mocks where coverage would otherwise plummet.
- Enforce import ordering via Ruff (`I` rule) and run `ruff check --fix` in CI to keep blocks alphabetized per AGENTS contract.
- Introduce a pytest fixture (or `monkeypatch` helper) that saves/restores `os.environ`, replacing the global mutation in `tests/conftest.py`.
- Document the provider abstraction, retry policy, and new schema requirements in CONTRIBUTING/setup docs so contributors follow the same patterns.

## Acceptance Criteria

- Supabase/OpenAI interactions recover automatically from transient failures, and per-taxonomy DSPy errors no longer terminate batch runs.
- Matching data captures semantic vs. LLM metrics separately, allows historical auditing, and exports expose clear match/not-match indicators.
- Embedding/LLM provider swaps require only configuration changes; embeddings are generated/stored in batches.
- Workflow batch files, polling, and logs include run ids and clean up after themselves.
- CI installs all required extras, coverage enforces the intended layers, imports stay ordered, and tests leave no environment residue.
<!-- 5ac5f2b8-85f1-4aad-9efd-5c5c5536fb9e 9340e0f9-1c82-4dac-9a6a-5de34b584d09 -->
# Workflow & Matching Hardening Plan

## Overview

Implement the five architecture changes identified during the review: persist embeddings in Supabase, enforce one canonical match per taxonomy row, shrink the LLM fallback search space, batch database writes during ingestion, and make the full workflow resumable via persisted “run” metadata. The goal is to cut token spend, reduce latency, and make large runs more reliable.

## Implementation Steps

### 1. Persist Embeddings & Move Similarity Search Into Supabase

**Files:** `src/data/schema.sql`, `src/data/supabase_client.py`, `src/services/matching.py`, `src/services/ingestion.py`, `src/services/categorization.py`, `docs/SETUP.md`

- Add `embedding vector(1536)` columns (or a dedicated `content_embeddings` table) plus `ivfflat` indexes for both `wordpress_content` and `taxonomy_pages`.
- Extend Supabase client with helpers to upsert embeddings and to run `match_documents` queries using pgvector `<->` ordering.
- Update ingestion + taxonomy loaders to call the embeddings API exactly once per row and persist vectors.
- Refactor `MatchingService` so the semantic step queries Supabase for top-K candidates using `similarity_threshold`, falling back to API calls only when missing embeddings.
- Document the pgvector requirement (extension enabled, index creation) in setup instructions.

### 2. Enforce One Canonical Match Record Per Taxonomy

**Files:** `src/data/schema.sql`, `src/data/supabase_client.py`, `src/services/matching.py`, `src/exporters/csv_exporter.py`, `tests/**`

- Introduce a unique constraint on `matching_results.taxonomy_id` (or replace the table with `taxonomy_matches` where each row stores the best current match and stage metadata).
- Update persistence paths to upsert by `taxonomy_id` only, ensuring previous matches get overwritten.
- Replace `get_unmatched_taxonomy` map/dict hacks with a direct SQL filter (e.g., `WHERE similarity_score < threshold OR content_id IS NULL`).
- Adjust exporter/analytics code to rely on the canonical row.
- Backfill tests to cover duplicate prevention and stage transitions.

### 3. Limit LLM/DSPy Fallback Inputs to Top Semantic Candidates

**Files:** `src/services/workflow.py`, `src/services/categorization.py`, `src/optimization/dspy_optimizer.py`, `docs/DSPY_IMPLEMENTATION.md`

- After stage 1, fetch only the top N (configurable, default 10) semantic candidates per taxonomy and pass that shortlist to `categorize_for_matching`.
- Update DSPy optimizer utilities so `_format_content_summaries` accepts the filtered list and so `predict_match` enforces the cap.
- Add configuration knobs (`LLM_CANDIDATE_LIMIT`, `LLM_CANDIDATE_MIN_SCORE`) and document their tuning guidance.
- Write regression tests showing token count reductions (mocked) and ensuring fallback still functions when candidate pools are small.

### 4. Batch Database Writes During Ingestion & Matching

**Files:** `src/services/ingestion.py`, `src/data/supabase_client.py`, `src/services/matching.py`, `tests/unit/test_ingestion.py`

- Add `bulk_upsert_content`, `bulk_upsert_taxonomy`, and `bulk_upsert_matchings` utilities that group rows in chunks (e.g., 200) and issue a single Supabase RPC per chunk.
- Update ingestion loops to collect rows before upserting, falling back to single-row writes when chunked calls fail.
- Apply the same batching to semantic + LLM stage persistence so each matching round commits in batches.
- Provide feature flags/env vars to tune batch sizes without code edits.

### 5. Persist Workflow Runs & Support Resumable Execution

**Files:** `src/data/schema.sql`, `src/services/workflow.py`, `src/cli.py`, `src/models.py`, `src/data/supabase_client.py`, `docs/QUICKSTART.md`

- Add a `workflow_runs` table capturing run id, stage, config overrides, timestamps, and counts.
- Expose CLI commands (`workflow start`, `workflow resume`, `workflow status`) that create/update run records and invoke sub-commands (load taxonomy, ingest, match, export) idempotently.
- Teach `WorkflowService` to checkpoint after each major stage and to skip completed stages when resuming.
- Emit structured logs tied to `run_id` so analysts can trace progress across retries.
- Document the new workflow commands and describe how to monitor/resume long jobs.

## Non-Functional Considerations

- Ensure all new SQL migrations are backward compatible (e.g., default embeddings as NULL until recomputed).
- Provide a one-time migration script to backfill stored embeddings for existing content to avoid cold-start latency during matching.
- Update observability/logging to surface batch sizes, candidate counts, and run ids for debugging.

### To-dos

- [ ] Add pgvector-powered embedding storage plus Supabase helpers and refresh existing records
- [ ] Enforce single-row canonical matches per taxonomy and update dependent services/tests
- [ ] Limit DSPy/LLM fallback prompts to top-N semantic candidates with new config knobs
- [ ] Implement bulk upsert paths for ingestion + matching persistence, configurable batch sizes
- [ ] Create `workflow_runs` persistence plus CLI commands for start/resume/status with documentation updates

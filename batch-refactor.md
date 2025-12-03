# LLM Fallback → OpenAI Batch API Migration

We will remove synchronous chat completions for the LLM fallback and use **only** the OpenAI Batch API to process all unmatched content items. This reduces cost, avoids per-request rate limits, and keeps the rubric gating intact.

## Objectives
- Replace synchronous DSPy/chat calls with a Batch-only pipeline for the LLM fallback stage.
- Keep the content→taxonomy data model intact (`matching_results` per `content_id`).
- Preserve rubric gating and semantic evidence when persisting results.
- Provide operational commands and a clear runbook.

## Key Files to Touch
- `src/services/categorization.py` — replace per-item calls with batch build/submit/poll/parse.
- `src/services/workflow.py` — LLM stage calls batch pipeline (no sync fallback).
- `src/cli.py` — add CLI commands for batch submit/status/apply; remove sync path.
- `src/data/supabase_client.py` — ensure upsert semantics remain content-first; may add helpers for staging batch outputs if needed.
- `docs/SETUP.md`, `README.md` — document batch-only flow and commands.

## Plan / Tasks

- [x] **Batch input builder**
  - Gather unmatched content (semantic < threshold).
  - Build JSONL lines per OpenAI Batch spec:
    ```json
    {
      "custom_id": "<content_id>",
      "method": "POST",
      "url": "/v1/chat/completions",
      "body": {
        "model": "<LLM_MODEL>",
        "messages": [...rubric prompt with candidate summaries...],
        "temperature": 0.0
      }
    }
    ```
  - Write to `data/batch/match_llm_requests.jsonl` (chunk if >5k records).

- [x] **Batch submit + poll**
  - Upload JSONL: `client.files.create(purpose="batch")`.
  - Submit batch: `batches.create(input_file_id=..., endpoint="/v1/chat/completions", completion_window="24h")`.
  - Poll `batches.retrieve` until `status == "completed"` (respect `LLM_BATCH_TIMEOUT`).
  - Download results: `files.content(output_file_id)`.

- [x] **Parse + persist**
  - Parse each output line, extract chosen taxonomy + rubric scores/decision.
  - Upsert `matching_results` (content-first):
    - `taxonomy_id` set on accept; otherwise `match_stage=needs_human_review`.
    - Keep `semantic_taxonomy_id` / `semantic_similarity_score`.
    - Store `llm_topic_score` and rubric blob; set `failed_at_stage="llm_batch"` on rejects.

- [x] **Workflow integration**
  - In `WorkflowService`, LLM stage always runs batch pipeline for all semantic-miss content.
  - Remove/disable synchronous DSPy per-item path.

- [x] **CLI / Ops**
  - `match` command triggers batch flow automatically.
  - Add maintenance commands:
    - `batch submit` (optional explicit control)
    - `batch status --id <batch_id>`
    - `batch apply --id <batch_id>` (parse & persist results)

- [x] **Tests**
  - Unit: JSONL builder (schema, chunking).
  - Unit: Batch output parser → `MatchingResult`.
  - Workflow unit: ensure LLM stage invokes batch path.
  - CLI unit: dry-run builder without submit.

- [x] **Docs**
  - Update README/SETUP with batch-only flow, commands, cost notes, and 24h SLA.
  - Include runbook: build → submit → poll → apply.

## Open Questions / Assumptions
- Batch size limits: assume up to ~10k requests per batch (default OpenAI guidance); chunk if larger.
- Cost model: use `gpt-4o-mini` unless we choose a cheaper model for batch.
- Rubric prompt: reuse current DSPy rubric text; keep temperature 0.0 for determinism.

## Definition of Done
- LLM fallback uses only Batch API.
- Matching results upsert cleanly (unique `content_id`) with rubric metadata preserved.
- CLI/docs updated; tests pass.

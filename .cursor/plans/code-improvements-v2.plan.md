<!-- 5ac5f2b8-85f1-4aad-9efd-5c5c5536fb9e 5d4f7c97-7e6b-46e5-9e90-6e2a2f3a8a11 -->
# Rubric-Gated Matching Hardening Plan (v2)

## Overview

Extend the rubric-gated LLM fallback we’ve implemented by: (1) clearly separating the selector vs. judge responsibilities in the matching stack, (2) tightening evaluation & promotion mechanics, (3) improving rubric observability, and (4) fully removing legacy confidence-based logic. The goal is to make LLM decisions more interpretable, easier to tune, and safer to promote, while keeping DSPy/GEPA focused on the parts of the prompt they optimize best.

## Implementation Steps

### 1. Clean Removal of Confidence-Based Logic

**Files:** `src/config.py`, `src/models.py`, `src/services/workflow.py`, `src/services/categorization.py`, `src/optimization/dspy_optimizer.py`, `src/optimization/evaluator.py`, `src/exporters/csv_exporter.py`, `tests/**`, `docs/OPTIMIZATION_QUICKSTART.md`, `docs/DSPY_IMPLEMENTATION.md`, `README.md`

- Remove `llm_confidence_threshold` from `Settings`:
- Delete the field and its validator in `src/config.py`.
- Strip confidence threshold mentions from CLI messages in `src/cli.py` and from `workflow.py` descriptions.
- Remove confidence from runtime models:
- Drop `confidence` from `CategorizationResult` and any other Pydantic models that no longer need it as a first-class field.
- Update all call sites (categorization, exporter) to stop reading or writing confidence.
- Strip confidence from DSPy dataset plumbing:
- In `DSPyOptimizer.load_training_dataset()` stop reading/storing `confidence` from CSV/JSON rows; the dataset still carries a column for legacy reasons but the DSPy `Example` should only include the fields the signature needs (taxonomy fields, summaries, best_match_index, optional rubric).
- Remove the confidence-penalty term from `accuracy_metric()` so evaluation purely reflects index correctness.
- Remove confidence-based evaluation and metrics:
- In `src/optimization/evaluator.py`, delete confidence fields from the aggregate stats structure and any references in logs.
- In `src/exporters/csv_exporter.py`, stop emitting `confidence` columns to exports (unless they’re purely historical and explicitly documented as such).
- Update docs to match:
- `docs/OPTIMIZATION_QUICKSTART.md`, `docs/DSPY_IMPLEMENTATION.md`, and `README.md` should no longer reference confidence thresholds; instead they should point at rubric thresholds and rubric acceptance rules.
- Ensure configuration tables list rubric settings (`llm_rubric_*`) and consensus knobs instead of the old threshold.
- Tests:
- Remove or update unit tests that assert on confidence threshold behaviour.
- Add regression tests that confirm the stack works end-to-end with no reliance on confidence.

### 2. Separate Selector vs. Judge Responsibilities

**Files:** `src/optimization/dspy_optimizer.py`, `src/services/categorization.py`, `src/services/matching.py`, `src/config.py`, `tests/unit/test_categorization.py`, `docs/DSPY_IMPLEMENTATION.md`

- Make the DSPy “selector” focused on ranking and index choice:
- Keep `TaxonomyMatcher` primarily responsible for choosing `best_match_index` from candidate summaries; rubric-style reasoning can remain, but is treated as *selector hints*, not as the final gate.
- Ensure the GEPA optimization metric remains tied to `best_match_index` correctness, not the rubric’s absolute values.
- Introduce or clarify a separate “judge” path:
- In `CategorizationService`, abstract the rubric evaluation into a dedicated method (or module) that:
- Takes a taxonomy + single best candidate summary.
- Produces rubric scores (`topic_alignment`, `intent_fit`, `entity_overlap`, `temporal_relevance`) and a `decision`.
- Option 1 (minimal change): reuse the DSPy matcher and call it in a “single-candidate judge mode” with an adjusted prompt that emphasizes scoring over selection.
- Option 2 (cleaner separation): introduce a lightweight non-DSPy prompt (plain OpenAI call) dedicated to rubric scoring; DSPy remains selector-only.
- Pick one approach and document it explicitly in `docs/DSPY_IMPLEMENTATION.md` so future changes don’t blur the boundary again.
- Wire selector → judge in the workflow:
- In `CategorizationService._find_best_match_llm`, first call the selector (DSPy) to get `best_match_index`, then call the judge to obtain rubric scores for that content item.
- Feed the judge’s rubric into `_accept_by_rubric()` for deterministic gating; selector outputs alone must not trigger acceptance.
- Tests:
- Add tests that:
- Mock the selector to return an index and the judge to return rubric scores, asserting that acceptance depends on the judge only.
- Verify that selector errors or low rubric scores fall back to human review, not semantic-only matches.

### 3. Robust Evaluation Harness for Matchers and Gating

**Files:** `src/optimization/evaluator.py` (or new module), `src/cli.py`, `src/data/supabase_client.py`, `src/services/workflow.py`, `docs/OPTIMIZATION_QUICKSTART.md`, `docs/DSPY_IMPLEMENTATION.md`, `tests/unit/test_evaluator.py`

- Add an explicit “matcher evaluation” CLI:
- New command: `python -m src.cli evaluate-matcher` with options:
- `--matcher-path` (optional, defaults to `matcher_latest.json`).
- `--dataset data/dspy_training_dataset.csv` or a Supabase-backed eval set selector.
- Flags for rubric vs. plain accuracy reporting.
- This command should:
- Load the specified matcher.
- Replay a held-out dataset through the selector and judge (including rubric gating).
- Emit summary metrics: exact-match accuracy, rubric acceptance rate, rejection rate, and any category-wise breakdowns.
- End-to-end workflow replay mode:
- Extend `Evaluator` or add a simple harness that:
- Connects to Supabase, pulls a slice of taxonomy/content, and runs the full cascade (semantic + LLM rubric gate).
- Computes: proportion auto-matched, rubric rejects, and human-review backlog under the current thresholds.
- Wire this into `src/cli.py` as a “compare matchers” command:
- Run the cascade with `matcher_latest.json`.
- Run again with a specified `matcher_vN.json`.
- Print a diff of key metrics to inform promotion decisions.
- Documentation:
- In `docs/DSPY_IMPLEMENTATION.md`, describe the evaluation pipeline and make clear which metrics are used to judge selector quality vs. rubric gate performance.
- Add a short “How to compare two matcher versions” section to `docs/OPTIMIZATION_QUICKSTART.md`.

### 4. More Structured Promotion and Version Control

**Files:** `scripts/promote_optimized_model.py`, `src/optimization/dspy_optimizer.py`, `prompt-optimiser/AGENTS.md`, `docs/OPTIMIZATION_QUICKSTART.md`, `README.md`

- Promote by explicit version:
- Extend `scripts/promote_optimized_model.py` to accept an optional `--version N` flag:
- If provided, promote `matcher_vN.json` instead of “latest”.
- If omitted, keep current behaviour (promote highest numbered version).
- On promotion, write a small manifest next to `matcher_latest.json` (e.g. `matcher_latest.meta.json`) containing:
- `version`, `validation_score`, `dataset_path`, `budget`, and timestamp.
- Enforce rubric compatibility at promotion time:
- In `DSPyOptimizer.promote_latest_model()`, verify that the candidate model includes rubric fields (e.g. by loading it and checking for `topic_alignment` et al.), and abort with a clear log if it’s confidence-only.
- Reflect this behaviour in the CLI script so users see an explicit error if they try to promote a legacy matcher.
- Documentation:
- Update `docs/OPTIMIZATION_QUICKSTART.md` and `README.md` to:
- Show usage examples for `--version`.
- Mention the meta file and how to interpret it when auditing promotion history.

### 5. Rubric Persistence and Observability

**Files:** `src/data/schema.sql`, `src/data/supabase_client.py`, `src/models.py`, `src/services/categorization.py`, `src/optimization/evaluator.py`, `docs/DSPY_IMPLEMENTATION.md`, `docs/SETUP.md`, `tests/unit/test_categorization.py`

- Persist rubric per match:
- Add a `rubric` JSON/JSONB column to `matching_results` (or a related `matching_rubrics` table keyed by matching id) in `src/data/schema.sql`.
- Extend Supabase client methods to insert/update the rubric blob whenever an LLM-based match is created or updated.
- Update `CategorizationService` so:
- On accept: it persists the full rubric dict along with the match.
- On reject/abstain: it either writes a partial rubric or leaves the field empty but logs the scores for diagnostics.
- Add rubric-aware evaluation summaries:
- In the evaluation module, compute distributions for topic/intent/entity/temporal scores among accepted vs. rejected matches.
- Emit aggregate stats (e.g. average topic score of accepted matches, percentage of rejects due to entity threshold) to help tune rubric thresholds.
- Documentation:
- In `docs/DSPY_IMPLEMENTATION.md`, describe how rubric data is stored and how analysts can query it (e.g. “find all matches where decision=reject but topic_alignment>0.9”).
- Note any Supabase requirements (JSONB, indexing on rubric fields if needed) in `docs/SETUP.md`.
- Database reset / migration script:
- Provide a lightweight SQL migration file or Python script that:
- Applies the rubric column addition safely to an existing database.
- Optionally truncates test data tables and reseeds them for local development.
- For local-only environments, add a `scripts/reset_supabase_local.py` helper that:
- Drops/recreates the schema via `src/data/schema.sql`.
- Clears matching/categorization tables so evaluation runs start from a clean slate.

## Non-Functional Considerations

- All schema changes should be additive (new columns/tables) and safe for existing data; destructive resets should be confined to explicit local-only scripts.
- The runtime path for semantic-only matches must remain fast; rubric persistence should only affect LLM fallback paths.
- Rubric evaluation should remain deterministic and cheap enough that consensus modes do not explode token usage (document recommended `llm_consensus_votes` for production vs. experiments).

### To-dos

- [ ] Remove confidence-based configuration, metrics, and usage from code and docs; rely solely on rubric gating.
- [ ] Separate selector (DSPy/GEPA-tuned) and judge (rubric scorer) pathways, with clear interfaces and tests.
- [ ] Add CLI/evaluator support to compare matcher versions using real cascade metrics (auto-accept vs. human-review rates).
- [ ] Enhance promotion tooling to support `--version`, write meta manifests, and enforce rubric-compatible model promotion.
- [ ] Extend schema and services to persist rubric JSON for each matched taxonomy row; add summary stats for rubric dimensions.
- [ ] Provide a small migration/reset script for local Supabase and update SETUP/implementation docs accordingly.
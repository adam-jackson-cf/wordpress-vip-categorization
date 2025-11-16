<!-- 91c4fbc6-e9e7-49df-8c86-aff880da5d74 0bbcc830-f3c1-4875-a648-9bc467844be6 -->
# Deterministic LLM Fallback Gating

## Context recap

- LLM stage is a fallback for nuanced cases only; semantic cosine already filtered/ordered candidates.
- The current acceptance uses an uncalibrated, self-reported LLM `confidence ≥ 0.9`.
- We will keep LLM as the selector for nuance, but make acceptance deterministic and auditable without over-relying on cosine again.

## Design

- Keep LLM’s role: choose `best_match_index` among pre-filtered candidates.
- Replace single `confidence` with a structured rubric the LLM scores deterministically:
  - topic_alignment: float 0–1 (does content cover the taxonomy topic?)
  - intent_fit: float 0–1 (does the content purpose match the taxonomy intent?)
  - entity_overlap: float 0–1 (named entities/keywords alignment; 0 if taxonomy has none)
  - temporal_relevance: float 0–1 (if taxonomy implies recency; else mark n/a)
  - decision: enum {accept,reject,abstain}
  - reasoning: short string
- Deterministic aggregator on our side:
  - Accept only if decision == accept AND topic_alignment ≥ T1 AND intent_fit ≥ T2 AND (if provided) entity_overlap ≥ T3.
  - If any criterion is n/a, it is ignored.
  - Persist per-criterion scores for audit.
- DSPy/GEPA prompts remain responsible for producing rubric-complete outputs; optimization now tunes rubric phrasing/demos.
- Temperatures and variance:
  - Keep DSPy LM temp moderately low (0.2–0.3) to retain nuanced evaluation.
  - Add optional `llm_consensus_votes` (default 1). If >1, run the rubric k times with different seeds and accept only on majority consensus (precision-oriented).
  - Avoid re-using cosine as an acceptance gate (we already used it to shortlist). Only sanity check: chosen item must be within the provided candidate list (already true) and not obviously off-topic (covered by rubric scores).

## Dataset alignment

- Extend `scripts/generate_dspy_dataset.py` so each example includes rubric metadata:
  - fields like `rubric_topic`, `rubric_intent`, `rubric_entity`, `rubric_temporal`, `rubric_decision`.
  - deterministic heuristics determine when temporal/entity dimensions are applicable.
- Refresh `data/dspy_training_dataset.csv` (already 360 rows) with these new columns; update README/docs to call out the richer schema.
- Update any fixtures/tests that ingest the dataset to expect the new columns.

## DSPy / GEPA adjustments

- `TaxonomyMatcher` signature must emit rubric fields instead of bare confidence.
- GEPA evaluation metric should consider rubric accuracy (either treat rubric columns as auxiliary outputs or keep the scoring on `best_match_index` while ensuring rubric-compatible instructions). Clarify in code comments that DSPy still optimizes the selection, while rubric outputs feed downstream gating.
- Update optimization scripts (quick test + GEPA workflow) to expect the new signature/output artifacts.
- Ensure existing `matcher_vN` artifacts are treated as legacy; only rubric-compatible models should be promoted once this work lands (document minimum version and enforce via `promote_optimized_model.py` checks/logs).

## Changes

1. **Dataset & documentation**
   - Regenerate `data/dspy_training_dataset.csv` with rubric metadata.
   - Document the new columns in README/OPTIMIZATION_QUICKSTART and mention the 360-example baseline.
2. **Update matcher signature and prompt to output rubric fields instead of `confidence`.**
   - File: `src/optimization/dspy_optimizer.py`
   - Change `TaxonomyMatcher` outputs to include rubric fields noted above.
   - Update `_format_content_summaries` unchanged; keep model selection logic.
3. **Adapt `predict_match` to parse rubric outputs and still return `best_match_index` plus rubric dict.**
   - Return type: `(int, dict)`; dict contains topic_alignment, intent_fit, entity_overlap, temporal_relevance, decision, reasoning.
4. **Introduce deterministic gate in `CategorizationService`.**
   - File: `src/services/categorization.py`
   - New helper `_accept_by_rubric(r: dict) -> bool` comparing scores to thresholds.
   - Replace the `confidence >= min_confidence` check with the rubric-based decision; persist rubric fields (extend DB schema or store in an audit JSON blob).
5. **Configuration additions in `Settings`.**
   - File: `src/config.py`
   - Add: `llm_rubric_topic_min` (default 0.8), `llm_rubric_intent_min` (0.8), `llm_rubric_entity_min` (0.5), `llm_rubric_temporal_min` (0.5), `llm_consensus_votes` (1), `llm_match_temperature` (0.2).
   - Mark `llm_confidence_threshold` as deprecated in docs; keep for backward compatibility until legacy models are fully retired.
6. **Optional consensus implementation.**
   - If `llm_consensus_votes > 1`, call the rubric prompt multiple times with deterministic seed offsets and accept only on majority vote; average rubric scores for audit logging.
7. **Promotion safeguard.**
   - Update `scripts/promote_optimized_model.py` (or the CLI messaging) to warn when promoting a legacy `matcher_vN` that lacks rubric outputs; only promote versions tagged as rubric-ready.
8. **Tests and docs.**
   - Unit tests verifying rubric acceptance/rejection paths (including consensus flows).
   - Tests for dataset generator ensuring rubric columns are produced and non-empty.
   - Update README/docs to explain the deterministic rubric, new dataset schema, and promotion requirements.

## Notes on calibration (post-hoc, optional)

- Later, we can fit a calibration model (e.g., with scikit-learn’s `CalibratedClassifierCV` or isotonic regression) mapping rubric features to a probability of correctness using a labelled validation set.
- This doesn’t change runtime behaviour now; it gives us a principled threshold if/when we want it. No need to add the dependency yet.

### To-dos

- [ ] Extend dataset generator + CSV to include rubric metadata, document schema/360-row baseline.
- [ ] Change DSPy signature to output rubric fields (drop legacy confidence).
- [ ] Parse rubric fields in `predict_match` and return `(index, rubric dict)`.
- [ ] Implement rubric gating in `CategorizationService` replacing confidence check (persist rubric audit data).
- [ ] Add rubric thresholds, entity/temporal knobs, consensus settings to `Settings`.
- [ ] Support majority-vote rubric evaluations when `llm_consensus_votes > 1`.
- [ ] Enforce promotion of rubric-capable models only; warn on legacy artifacts.
- [ ] Add unit tests for dataset/rubric acceptance paths and update docs (README, OPTIMIZATION_QUICKSTART) with new behaviour.

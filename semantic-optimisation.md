# Semantic Optimisation Plan

## 1. Problem Statement

Recent semantic runs still leave **964/1000** content rows below the 0.70 cosine threshold even after we enforced regulatory gating (primary/secondary audiences + species). Compliance filters now prevent invalid matches, but they do not improve the vector similarity of compliant pairs, so we frequently end up with low-scoring leftovers. Root causes observed:

1. **Embedding weighting** – taxonomy vectors still treat the semantic summary as dominant; the priority fields (`Destination_URL`, `Local_Page_Name`, `Key_Topic`, then summary) are not emphasized, so URL/locale alignment gets lost.
2. **Sparse detections** – WordPress pages lacking clear veterinarian/producers/species phrasing yield empty detector sets, causing compliant taxonomy rows to be filtered out entirely.
3. **Content-side signals** – We still embed large plain-text blobs; URLs, slugs, headings, and key entities are not weighted, so compliance-aligned pairs can remain semantically “far”.
4. **Scoring feedback** – The DSPy/LLM fallbacks see detection cues, but we haven’t retrained or added scoring bonuses that explicitly reward compliant matches, leaving cosine scores stuck in the 0.6 range.

## 2. Goals

- Increase the proportion of compliant matches ≥0.70 by weighting the key taxonomy fields and aligning content embeddings with audience/species hints.
- Reduce “empty detection” failures through heuristic inference so the gate stops discarding viable rows.
- Produce diagnostics (report + metrics) that highlight where compliance filtering is suppressing scores.

## 3. Workstreams & Task List

### 3.1 Taxonomy Embedding Reweighting

- [ ] **Read context**: `src/services/matching.py:create_taxonomy_text`, `semantic-improvements.md` (ensure no conflicting guidance).
- [ ] **Update weighting**: in `create_taxonomy_text`, duplicate/annotate priority fields, e.g.:
  - Prepend `Destination URL Path` and `Local Page Name` blocks with emphasis strings such as `Priority Field:` (twice if needed).
  - Move `Key Topics` ahead of `Summary` and optionally repeat each topic once.
- [ ] **Tests**: expand `tests/unit/test_matching.py` to assert the new ordering/duplicates.

### 3.2 Content Embedding Enhancements

- [ ] **Fields audit**: inspect `src/services/matching.py:create_content_text` and confirm which metadata slots are populated (slug, categories, tags, detectors, preview).
- [ ] **Inject structured cues**:
  - Add `Destination-esque` tokens (URL path segments) with emphasis.
  - Include first 3 headings (if available in metadata) and highlight `Detected Audiences/Species` again to boost importance.
  - Consider truncating previews to 1k chars but duplicating the title/slug for weight.
- [ ] **Tests**: extend `tests/unit/test_matching.py` to cover new sections.

### 3.3 Detection Coverage Improvements

- [ ] **Analyze gaps**: run `scripts/generate_report.py` to quantify `primary_only empty_detection` counts; capture baseline in `results/semantic_match_analysis_<date>.md`.
- [ ] **Heuristic enrichment** (`src/services/detection.py` + `src/connectors/wordpress_vip.py`):
  - [ ] Expand `AUDIENCE_TERMS`/`SPECIES_TERMS` with domain-specific synonyms gleaned from Spain taxonomy (e.g., `ganaderia extensiva`, `aves acuaticas`).
  - [ ] Add site/URL-path heuristics (e.g., if URL contains `/tutores-de-mascotas/`, infer `pet owners`).
  - [ ] Store inferred values in `WordPressContent.metadata` with provenance (e.g., `inferred_from_path`).
- [ ] **Validation**: add unit tests in `tests/unit/test_detection.py` + `tests/unit/test_full_pipeline_mock.py` to ensure new synonyms trigger detections.

### 3.4 Semantic Scoring Adjustments

- [ ] **Priority boosts**: extend `_priority_boost` in `src/services/matching.py` to award larger bonuses when both taxonomy species and audience match (e.g., +0.05) or when URL tokens overlap.
- [ ] **DSPy retraining**: update `semantic-improvements.md` with a follow-on task to rerun `python -m src.cli optimize-dataset --optimizer gepa` after embedding tweaks.
- [ ] **LLM prompts**: ensure `src/services/categorization.py` references the reweighted cues (already includes detectors; verify that candidate summaries now highlight local names and topics).
- [ ] **Tests**: adjust `tests/unit/test_matching.py` for new boost amounts and `tests/unit/test_categorization.py` if prompt snippets change.

### 3.5 Diagnostics & Reporting

- [ ] **Report update**: enhance `scripts/generate_report.py` to include a table comparing “compliance-filtered but <0.70” rows before/after weighting.
- [ ] **Logging**: in `src/services/matching.py`, add debug logs when compliant candidates fall below threshold to aid smoke tests (ensure logging adheres to performance rules).
- [ ] **Documentation**: update `README.md` + `docs/SETUP.md` with guidance on CA-wrapped commands and the new semantic strategy; note that weighted embeddings are required before matching.

### 3.6 Validation & Rollout

- [ ] **Smoke tests**: `scripts/corp_ca_exec.sh uv run python -m src.cli match --skip-llm --limit 20` with logging, inspect new cosine distribution.
- [ ] **Full unit suite**: `uv run pytest tests/unit` (verify only the known third-party warning remains).
- [ ] **Integration smoke**: optional `scripts/corp_ca_exec.sh uv run pytest tests/integration/test_full_pipeline.py -k semantic --maxfail=1` if Supabase creds allow.
- [ ] **Update plans**: sync completion status back into `semantic-improvements.md` and this document.

## 4. Success Metrics

- ≥50% of compliant rows scoring ≥0.70 on a 1,000-row sample.
- Detection coverage: <10% of primary-only rows with `empty_detection` counts in the report.
- Reduced reliance on LLM fallback (LLM queue <5% of attempted matches in smoke test).

## 5. File Reference Summary

| Area | Files |
| --- | --- |
| Taxonomy weighting | `src/services/matching.py`, `tests/unit/test_matching.py` |
| Content cues | `src/services/matching.py`, `tests/unit/test_matching.py` |
| Detection | `src/services/detection.py`, `src/connectors/wordpress_vip.py`, `tests/unit/test_detection.py`, `tests/unit/test_full_pipeline_mock.py` |
| Scoring & DSPy | `src/services/matching.py`, `src/optimization/dspy_optimizer.py`, `semantic-improvements.md` |
| Reporting | `scripts/generate_report.py`, `README.md`, `docs/SETUP.md` |
| Validation | `tests/unit/*`, `tests/integration/*`, CLI commands |

# Semantic Matching Compliance Roadmap

## Objective
Design and deliver an audience/species-aware semantic matching workflow that satisfies regulatory constraints without relying on REST-provided `target_audience` or `species` fields (which are mostly empty in production). We will:
1. Detect audience/species cues ourselves from WordPress content.
2. Persist those cues alongside embeddings.
3. Weight/gate semantic candidates based on the detected cues.
4. Surface compliance signals through the LLM fallback and reporting layers.

All work will land directly on `main`. Treat this document as the implementation playbook.

---

## 1. Research & Context Files
Before coding, read/review:
- `src/connectors/wordpress_vip.py`: Understand the existing metadata extraction path so we can add on-the-fly detectors.
- `src/services/ingestion.py`: See where content is fetched, transformed, and upserted (we’ll inject detection results here).
- `src/services/matching.py`: Identify where to hook gating/scoring logic.
- `src/data/schema.sql` + `src/models.py`: Know the schema/model shape before adding columns.
- `src/services/categorization.py`, `tests/unit/test_categorization.py`: Ensure LLM prompts can leverage new signals.
- `scripts/generate_report.py`: Determine where to add reporting metrics.

---

## 2. Work Plan

### 2.1 Detection Dictionaries & Helper Module
- Create `src/services/detection.py` (or similar) containing:
  - `AUDIENCE_TERMS`: mapping canonical tokens (`"veterinarians"`, `"producers"`, `"pet_owners"`, etc.) to multilingual keyword lists.
  - `SPECIES_TERMS`: mapping taxonomy species tokens (`"swine"`, `"bovine"`, `"poultry"`, etc.) to synonyms.
  - Helper functions:
    - `detect_audiences(text: str) -> set[str]`
    - `detect_species(text: str) -> set[str]`
  - Keep dictionaries plain JSON-friendly to allow future config loading.
- Tests: `tests/unit/test_detection.py` verifying that Spanish/English phrases are recognized and that noise words aren’t flagged.

### 2.2 Persist Detection Results
- Update `wordpress_content` schema (`src/data/schema.sql`) to add:
  - `detected_audiences JSONB DEFAULT '[]'`
  - `detected_species JSONB DEFAULT '[]'`
- Mirror fields in `WordPressContent` model + Supabase client.
- Enhance `WordPressVIPConnector._parse_wordpress_item` (or ingestion stage) to:
  - Run the detector on `title`, `excerpt`, first N characters of `content`.
  - Attach normalized sets to `WordPressContent.metadata` and new columns.
- Update ingestion tests (`tests/unit/test_ingestion.py`, `tests/unit/test_full_pipeline_mock.py`) to assert detectors run and values persist.

### 2.3 Embedding Text Adjustments
- Modify `MatchingService.create_content_text` / `create_taxonomy_text` to prepend sections:
  - `Audience Hint: veterinarians` (list all taxonomy audiences or detections)
  - `Species Hint: swine`
- Ensure these lines are present even when detection fails (use `unknown`) so embeddings learn the absence.
- Add tests verifying the new lines appear (`tests/unit/test_matching.py`).

### 2.4 Matching Gates + Scoring
- In `MatchingService`:
  - Load `WordPressContent.detected_audiences/detected_species` (fallback to metadata) for gating.
  - Implement compliance helpers (similar to current branch):
    - `audience_required`: if taxonomy has only primary, enforce exact match; if primary+secondary, allow either; wildcard when none.
    - `species_required`: taxonomy species must be subset of detected species.
  - Apply gating just after retrieving pgvector results. If everything is filtered out, log + mark for manual review.
  - Apply a small boost to cosine scores when both gates pass to break ties.
- Update `tests/unit/test_matching.py` to cover:
  - Audience mismatch yields no candidates (primary-only case).
  - Species mismatch yields no candidates.
  - Boost path increases score (use `pytest.approx`).

### 2.5 Supabase Flow
- Modify `src/data/supabase_client.py` to persist new columns on upsert/bulk operations.
- Update relevant Supabase unit tests to assert JSONB sets round-trip.

### 2.6 LLM + Reporting
- `src/services/categorization.py` prompts: include detected audience/species for context so the judge can reject mismatched pages.
- `tests/unit/test_categorization.py`: ensure prompts mention the new fields.
- `scripts/generate_report.py`: add metrics summarizing how often detections align with taxonomy expectations (e.g., % of semantic matches with matching audiences). Note: this may require a migration step to backfill detections for existing rows.

### 2.7 Tooling & Docs
- `docs/SETUP.md` + `README.md`: document the new schema fields, detection logic, and compliance behavior.
- `data/taxonomy.csv.example`: highlight species/audience columns’ role.
- `semantic-improvements.md`: keep this plan updated with progress (use checkboxes, etc.).

### 2.8 Validation Checklist
- Run targeted unit tests:
  - `uv run pytest tests/unit/test_detection.py tests/unit/test_matching.py tests/unit/test_ingestion.py tests/unit/test_full_pipeline_mock.py`
- Run broader suite: `scripts/corp_ca_exec.sh uv run pytest tests/unit` for a final check.
- If database migrations are needed, update `docs/SETUP.md` with the SQL diff and remind ops to apply before deploying.

---

## 3. Open Questions / Future Enhancements
- Should detection dictionaries be configurable via JSON files (for non-code updates)?
- Do we need to backfill existing `wordpress_content` rows with detection results (via a one-off script)?
- Explore optional `mahAnalytics` parsing only if REST metadata remains empty for key sections (future iteration).

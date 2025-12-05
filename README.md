# WordPress VIP Content Categorization

AI-powered workflow that ingests content from WordPress VIP, stores it in Supabase, and runs a cascading semantic → LLM workflow to map every WordPress content item to the best matching taxonomy destination.

## System Workflow

1. **URL reference matching (stage 0)** – Before embeddings run, content whose metadata categories intersect `URL_CHECKER_CATEGORY_IDS` (e.g., `274` for Spain “Noticias”) must first match a taxonomy `reference_source`. Exact path hits are auto-accepted (`match_stage=url_matching`, score `1.0`) and skip downstream stages; misses are recorded as `match_stage=url_checker_excluded` / `failed_at_stage=url_check_excluded`, preventing those press releases from entering the semantic/LLM queues until a taxonomy reference is added.
2. **Semantic matching (default ≥0.70)** – Embed content + taxonomy (OpenAI `text-embedding-3-small` by default), compute cosine similarity, and persist the top taxonomy candidate per content row.
3. **LLM batch fallback (rubric-gated)** – For any content whose semantic score falls below the floor, enqueue the items via the OpenAI Batch API (chat completions) so the rubric-judged fallback runs asynchronously. The pipeline automatically waits for completion during `match` runs, but you can also submit/poll/apply batches manually. The Batch prompt is sourced from the latest DSPy/GEPA-optimized matcher (instructions + demonstrations), so rerunning the optimizer and promoting the artifact immediately updates production prompts.
4. **Human review** – Export content rows that still lack an accepted taxonomy so analysts can triage the backlog.

Toggle any stage with `ENABLE_SEMANTIC_MATCHING`, `ENABLE_LLM_CATEGORIZATION`, the new `ENABLE_CONTENT_TYPE_HINTING` (turns the content-type bonus on/off), or `ENABLE_URL_STAGE_ZERO` (turns reference-source URL matching on/off). Adjust thresholds via `SIMILARITY_THRESHOLD` and rubric settings: `LLM_RUBRIC_TOPIC_MIN`, `LLM_RUBRIC_INTENT_MIN`, `LLM_RUBRIC_ENTITY_MIN`, optional `LLM_CONSENSUS_VOTES`.

### Regulatory Compliance Signals

- **Detector-backed metadata** – During ingestion every WordPress page runs through `src/services/detection.py`, storing multilingual `detected_audiences` / `detected_species` sets in both the Supabase columns and the JSON metadata. These cues are always embedded in the content vectors (`MatchingService.create_content_text`).
- **Content-type hints** – Structural hints (catalogue pages, news hubs, legal content, etc.) now live in `data/detection_content_types.json`. Each taxonomy `content_type` has bilingual keywords (Spanish + English) so WordPress slugs in any language still trigger the rule; add both variants whenever you onboard a new market.
- **Taxonomy cues** – `create_taxonomy_text` now emits explicit `Primary Audience`, `Secondary Audience`, and `Species` lines even when the fields are blank so embeddings can learn the absence of a secondary audience (which means “primary-only” compliance).
- **Semantic gating** – Before any cosine candidate survives, the matching service enforces the regulatory rules: taxonomy rows with only a primary audience require an exact detector match, dual-audience rows accept either, and species lists must be a subset of detected species. Compliant pairs get a micro boost so they float to the top of the candidate list.
- **LLM visibility** – The LLM fallback prompt now includes the detector output in the content section, so rubric decisions can explicitly reason about audience/species alignment instead of inferring it from the raw article.
- **Reporting** – `scripts/generate_report.py` summarizes how often detector outputs align with taxonomy rules (primary-only coverage, dual-audience coverage, species compliance) so we can trend adherence before/after each tuning cycle.

## Operations Quick Reference

- `python -m src.cli full-run --output results/results.csv` – Taxonomy load → ingestion → cascading matching → CSV export.
- `python -m src.cli ingest --resume` – Incremental ingestion (override with `--since YYYY-MM-DD`).
- `python -m src.cli match --only-unmatched --skip-semantic --force-llm` – Re-drive backlog beneath the semantic threshold (builds + applies fresh batch jobs).
- `python -m src.cli batch submit --limit 50 --no-wait` – Submit queued `needs_llm_review` rows to OpenAI Batch without blocking.
- `python -m src.cli batch status --id batch_xxx` / `python -m src.cli batch apply --id batch_xxx` – Inspect or apply historical batch jobs if you skipped waiting during `match`.
- `python -m src.cli workflow start|resume|status` – Managed runs with persisted checkpoints.
- `python -m src.cli init-db` – Apply `src/data/schema.sql` to Supabase (RPC when available, otherwise prints SQL).
- `python scripts/test_setup.py` – Verifies env vars, Supabase access, WordPress connector, and embeddings.
- `scripts/bootstrap_supabase.py --run-tests` – Helper for provisioning Supabase plus smoke tests.

## Setup Paths

### Quick Start
Use [docs/QUICKSTART.md](docs/QUICKSTART.md) for a scripted path: seed Supabase, load taxonomy data, and run `full-run` with one command.

### Full Install & Local Development
Follow [docs/SETUP.md](docs/SETUP.md) for prerequisites, virtualenv management, `.env` population, CLI command recipes, troubleshooting, and cost controls.

### Configuration Keys

- Required: `SUPABASE_URL`, `SUPABASE_KEY`, `WORDPRESS_VIP_SITE_TOKENS` (comma-separated `site|token` list).
- Optional: `ENABLE_CORP_CA` + `CORP_CA_BUNDLE_PATH` – set `ENABLE_CORP_CA=1` to have the build/test workflow wrap every command via `scripts/corp_ca_exec.sh`, which exports the bundle into `SSL_CERT_FILE`, `REQUESTS_CA_BUNDLE`, etc. You can also run ad-hoc commands (e.g. `scripts/corp_ca_exec.sh python -m src.cli full-run ...`) to ensure the CA chain is applied.
- Semantic embedding provider: `SEMANTIC_API_KEY`, `SEMANTIC_BASE_URL`, `SEMANTIC_EMBEDDING_MODEL`.
- LLM categorization provider: `LLM_API_KEY`, `LLM_BASE_URL`, `LLM_MODEL`, `LLM_BATCH_TIMEOUT`, `LLM_BATCH_COMPLETION_WINDOW`, `LLM_BATCH_CHUNK_SIZE`, `LLM_BATCH_ARTIFACT_DIR`.
- Workflow tuning: `SIMILARITY_THRESHOLD`, `ENABLE_SEMANTIC_MATCHING`, `ENABLE_LLM_CATEGORIZATION`.
- URL gate tuning: `URL_CHECKER_CATEGORY_IDS` (comma-separated WordPress category IDs that must pass Stage‑0 reference matching; e.g., `274` for Spain press releases).
- LLM rubric gate: `LLM_RUBRIC_TOPIC_MIN`, `LLM_RUBRIC_INTENT_MIN`, `LLM_RUBRIC_ENTITY_MIN`, `LLM_CONSENSUS_VOTES`, `LLM_MATCH_TEMPERATURE` (0–1).
- Data inputs: `TAXONOMY_FILE_PATH` (default `./data/Spain_New.csv`).

Legacy `OPENAI_*` keys are supported but should be replaced with the semantic / LLM-specific variables above. `.env.example` documents every option.

## Prompt Optimization

Use the DSPy tooling when tuning categorization prompts:

- `python scripts/run_optimization_workflow.py --dataset data/dspy_training_dataset.csv` – Bootstrap check then GEPA optimization (asks for confirmation before the expensive stage).
- `python -m src.cli optimize-dataset --dataset data/dspy_training_dataset.csv --optimizer bootstrap` – Cheap, single-stage tuning.
- `python -m src.cli optimize-dataset --dataset data/dspy_training_dataset.csv --optimizer gepa --budget medium` – Thorough tuning with GEPA.

Deep dives live in [docs/OPTIMIZATION_QUICKSTART.md](docs/OPTIMIZATION_QUICKSTART.md), [docs/DSPY_IMPLEMENTATION.md](docs/DSPY_IMPLEMENTATION.md), and [docs/DSPY_GEPA_BEST_PRACTICES.md](docs/DSPY_GEPA_BEST_PRACTICES.md).

### DSPy Dataset

The default dataset (`data/dspy_training_dataset.csv`) now contains 360 labeled examples (60 per taxonomy content type). Each example injects a guaranteed positive match drawn from:

- **Live feeds** – `wordpress.org/news`, `developer.wordpress.org/news`, `make.wordpress.org/community`, `woocommerce.com`, etc.
- **Curated enterprise cases** – Additional WooCommerce and healthcare-focused stories to keep underrepresented taxonomies well covered.

Regenerate or grow the dataset any time:

```bash
python scripts/generate_dspy_dataset.py \
  --taxonomy data/Spain_New.csv \
  --output data/dspy_training_dataset.csv \
  --num-examples 360
```

Feel free to raise `--num-examples` for even more coverage; the generator automatically balances per category.

## Semantic Optimization

The semantic matching layer underwent significant optimization to improve compliant match detection from **11.9% → 73.2% success rate** (+61.3pp, 6.1x multiplier). This section documents the strategies and testing approaches that drive high-quality semantic matches.

### Detection Strategy

Detection metadata (audiences and species) is externalized in `data/detection_terms.json`, allowing term expansion without code changes:

- **Text-based detection**: Scans content for multilingual synonyms (e.g., `veterinario`, `ganadero`, `porcino`, `bovino`) and stores detected audiences/species in Supabase columns.
- **URL path inference**: When text signals are weak, the detector examines URL slugs for patterns like `/porcino/`, `/veterinaria/`, `/mascotas/` to infer regulatory context.
- **Adding new terms**: Extend `audience_terms`, `species_terms`, `url_path_patterns`, and the `content_type_terms` entries in `data/detection_terms.json`, then mirror those keywords (plus the desired bonus values) in `data/detection_content_types.json`. Always include both Spanish and English variants for new markets so slug/path detection remains reliable.

**Example**:
```json
{
  "audience_terms": {
    "veterinarians": ["veterinario", "veterinaria", "vet", "clinica veterinaria"]
  },
  "url_path_patterns": {
    "species": {
      "swine": ["porcino", "cerdo"]
    }
  }
}
```

Detection runs during ingestion (`src/services/detection.py`) and populates `detected_audiences` / `detected_species` columns, which are always included in content embeddings to boost compliance alignment.

### Embedding Strategy

Semantic vectors are constructed with **priority field weighting** to emphasize the most discriminative taxonomy and content attributes:

**Taxonomy embeddings** (`create_taxonomy_text` in `src/services/matching.py`) duplicate high-signal fields:
- **Destination URL** (2×): Path segments (`/porcino/`, `/veterinarios/`) appear twice with "Priority Field" markers
- **Local Page Name** (2×): Human-readable page titles (e.g., "Porcino Landing Page")
- **Key Topics** (2×): Core subject areas (e.g., "Swine Health", "Vaccination")
- **Primary/Secondary Audiences** (1×): Explicit audience labels with "Primary Audience:", "Secondary Audience:" prefixes (even when blank, to encode absence)
- **Species** (1×): Species lists with "Species:" prefix
- **Semantic Summary** (1× at end): Deprioritized by placing last in the concatenation

**Content embeddings** (`create_content_text`) follow symmetric structure:
- **URL slug tokens** (2×): Extracted path segments mirroring taxonomy URL logic
- **Page title** (2×): Concentrated semantic signal
- **Detected audiences/species** (1×): Externalized detector outputs
- **Supporting metadata** (1×): Excerpt, categories, tags, detected language, and published date
- **Content preview** (1× at end): First 1000 chars, deprioritized like taxonomy summaries

**Symmetric design rationale**: Both taxonomy and content embeddings place URL/name/topic fields first with duplication, ensuring cosine similarity rewards structural alignment (URL overlap, topic match) before semantic prose.

### Scoring Enhancements

The matching service applies multi-layer scoring to boost compliant pairs beyond raw cosine similarity:

1. **Base embeddings**: Cosine similarity between content and taxonomy vectors (threshold ≥0.70 by default)
2. **Compliance bonus (+0.05)**: Awarded when **both** audience and species constraints align:
   - Taxonomy primary-only → exact detector match required
   - Taxonomy dual-audience → either primary or secondary must match
   - Species lists must be subsets of detected species
3. **URL overlap bonus (+0.03)**: Triggered when >30% of URL tokens match between content slug and taxonomy destination (e.g., `/porcino/salud/` ↔ `/porcino/productos/`)
4. **Debug logging**: When compliant pairs score <0.70, the system logs the base score, applied bonuses, and final result to aid tuning.

**Example scoring**:
```
Base cosine: 0.67
+ Compliance bonus: +0.05 (audience AND species aligned)
+ URL overlap bonus: +0.03 (40% token match)
─────────────────────────
Final score: 0.75 → ACCEPTED
```

### Testing Semantic Improvements

To isolate the semantic layer and exclude LLM fallback noise:

**Skip LLM categorization**:
```bash
python -m src.cli match --skip-llm --only-unmatched
```

**Force embedding regeneration** (useful after detection_terms.json changes or embedding strategy updates):
```sql
-- Clear content embeddings
UPDATE wordpress_content SET content_embedding = NULL;

-- Clear taxonomy embeddings
UPDATE taxonomy_pages SET taxonomy_embedding = NULL;
```

Then rerun `python -m src.cli match` to rebuild vectors and observe new score distributions.

**Analyze compliance alignment**:
```bash
python scripts/generate_report.py
```

This produces `results/semantic_match_analysis_<timestamp>.md` with compliance breakdown (primary-only coverage, dual-audience coverage, species alignment, detection gaps) and score distribution histograms.

### Performance Results

The optimization cycle delivered substantial improvements in compliant match detection:

| Metric | Baseline | Optimized | Change |
|--------|----------|-----------|--------|
| **Semantic Success Rate** | 11.9% | 73.2% | +61.3pp |
| **Compliant Matches (≥0.70)** | 25 | 166 | +564% |
| **Success Rate Multiplier** | 1.0× | 6.1× | 6.1× |
| **Empty Detection Gaps** | 16.0% | 44.6% | +28.6pp (improved coverage) |

**Key observations**:
- Priority field duplication shifted cosine scores upward by ~0.05-0.10 for structurally aligned pairs
- Compliance + URL bonuses rescued 15-20% of near-threshold pairs (0.65-0.69 → 0.70+)
- Detection term expansion reduced false negatives but increased "empty detection" logging (expected tradeoff; detection gaps now trigger URL inference fallback)

**Detailed metrics**: See `results/semantic_match_analysis_BASELINE_20251204.md` (pre-optimization) and `results/semantic_match_analysis_OPTIMIZED_20251204.md` (post-optimization) for drill-down analysis.

## Developer Process

- Run `make quality-check` before committing; it wraps `black --check`, `ruff check`, `mypy src`, and `pytest --cov=src --cov-fail-under=80`.
- Install hooks via `pre-commit install` to ensure the gate runs on every commit.
- Unit tests mock all external I/O; integration tests live in `tests/integration/` and must be marked `@pytest.mark.integration`.
- Use `pytest --no-cov tests/unit/test_x.py` for focused loops, but rerun the full gate (with coverage) before merging.
- Database initialization lives in `src/data/schema.sql`; rerun `python -m src.cli init-db` when migrations land.
- See the relevant `AGENTS.md` files for coding rules, project structure expectations, and test behaviors per directory.

## Directory Guide

- `src/` – CLI entrypoints, services, connectors, configuration, DSPy modules.
- `tests/` – Unit + integration suites with shared fixtures.
- `data/` – Taxonomy CSVs, result exports, and Supabase artifacts.
- `docs/` – Detailed how-to guides referenced throughout this README.
- `prompt-optimiser/` – Versioned DSPy optimization outputs (models/configs/reports).

## Documentation Hub

- [docs/QUICKSTART.md](docs/QUICKSTART.md) – Scripted onboarding path.
- [docs/SETUP.md](docs/SETUP.md) – Environment, CLI, troubleshooting, and cost guidance.
- [docs/OPTIMIZATION_QUICKSTART.md](docs/OPTIMIZATION_QUICKSTART.md) – DSPy / GEPA workflows.
- [docs/DSPY_IMPLEMENTATION.md](docs/DSPY_IMPLEMENTATION.md) – Optimizer internals and extension hooks.
- [docs/DSPY_GEPA_BEST_PRACTICES.md](docs/DSPY_GEPA_BEST_PRACTICES.md) – Advanced tuning guidance.
- [src/data/schema.sql](src/data/schema.sql) – Canonical Supabase schema.

## Need Help?

Troubleshooting tips for Supabase auth, OpenAI quotas, low match quality, and general FAQs live in [docs/SETUP.md](docs/SETUP.md).

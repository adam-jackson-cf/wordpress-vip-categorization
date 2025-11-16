# WordPress VIP Content Categorization

AI-powered workflow that ingests content from WordPress VIP, stores it in Supabase, and runs a cascading semantic → LLM workflow to map taxonomy URLs to the best matching WordPress content.

## System Workflow

1. **Semantic matching (default ≥0.85)** – Embed taxonomy + content, compute cosine similarity, and write matches immediately.
2. **LLM categorization fallback (rubric-gated)** – Ask the configured chat model to choose the best remaining candidate per taxonomy page, then apply a deterministic rubric gate (topic/intent/entity thresholds) for precision-first acceptance.
3. **Human review** – Export unmapped taxonomy rows (blank targets) for analysts.

Toggle any stage with `ENABLE_SEMANTIC_MATCHING` / `ENABLE_LLM_CATEGORIZATION` or CLI flags; adjust thresholds via `SIMILARITY_THRESHOLD` and rubric settings: `LLM_RUBRIC_TOPIC_MIN`, `LLM_RUBRIC_INTENT_MIN`, `LLM_RUBRIC_ENTITY_MIN`, optional `LLM_CONSENSUS_VOTES`. `LLM_CONFIDENCE_THRESHOLD` is deprecated.

## Operations Quick Reference

- `python -m src.cli full-run --output results/results.csv` – Taxonomy load → ingestion → cascading matching → CSV export.
- `python -m src.cli ingest --resume` – Incremental ingestion (override with `--since YYYY-MM-DD`).
- `python -m src.cli match --only-unmatched --skip-semantic --force-llm` – Re-drive backlog beneath the semantic threshold.
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

- Required: `SUPABASE_URL`, `SUPABASE_KEY`, `WORDPRESS_VIP_SITES` (comma-separated list).
- Semantic embedding provider: `SEMANTIC_API_KEY`, `SEMANTIC_BASE_URL`, `SEMANTIC_EMBEDDING_MODEL`.
- LLM categorization provider: `LLM_API_KEY`, `LLM_BASE_URL`, `LLM_MODEL`, `LLM_BATCH_TIMEOUT`.
- Workflow tuning: `SIMILARITY_THRESHOLD`, `ENABLE_SEMANTIC_MATCHING`, `ENABLE_LLM_CATEGORIZATION`.
- LLM rubric gate: `LLM_RUBRIC_TOPIC_MIN`, `LLM_RUBRIC_INTENT_MIN`, `LLM_RUBRIC_ENTITY_MIN`, `LLM_CONSENSUS_VOTES`, `LLM_MATCH_TEMPERATURE` (0–1). `LLM_CONFIDENCE_THRESHOLD` (deprecated).
- Data inputs: `TAXONOMY_FILE_PATH` (default `./data/taxonomy.csv`).

Legacy `OPENAI_*` keys are supported but should be replaced with the semantic / LLM-specific variables above. `.env.example` documents every option.

## Prompt Optimization

Use the DSPy tooling when tuning categorization prompts:

- `python scripts/run_optimization_workflow.py --dataset data/dspy_training_dataset.csv` – Bootstrap check then GEPA optimization (asks for confirmation before the expensive stage).
- `python -m src.cli optimize-dataset --dataset data/dspy_training_dataset.csv --optimizer bootstrap` – Cheap, single-stage tuning.
- `python -m src.cli optimize-dataset --dataset data/dspy_training_dataset.csv --optimizer gepa --budget medium` – Thorough tuning with GEPA.

Deep dives live in [docs/OPTIMIZATION_QUICKSTART.md](docs/OPTIMIZATION_QUICKSTART.md), [docs/DSPY_IMPLEMENTATION.md](docs/DSPY_IMPLEMENTATION.md), and [docs/DSPY_GEPA_BEST_PRACTICES.md](docs/DSPY_GEPA_BEST_PRACTICES.md).

### DSPy Dataset

The default dataset (`data/dspy_training_dataset.csv`) now contains 360 labeled examples (60 per taxonomy category). Each example injects a guaranteed positive match drawn from:

- **Live feeds** – `wordpress.org/news`, `developer.wordpress.org/news`, `make.wordpress.org/community`, `woocommerce.com`, etc.
- **Curated enterprise cases** – Additional WooCommerce and healthcare-focused stories to keep underrepresented taxonomies well covered.

Regenerate or grow the dataset any time:

```bash
python scripts/generate_dspy_dataset.py \
  --taxonomy data/taxonomy.csv \
  --output data/dspy_training_dataset.csv \
  --num-examples 360
```

Feel free to raise `--num-examples` for even more coverage; the generator automatically balances per category.

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
- [CODE_REVIEW.md](CODE_REVIEW.md) & [docs/CODE_REVIEW_GUIDE.md](docs/CODE_REVIEW_GUIDE.md) – Review checklist plus AI/human workflow.
- [src/data/schema.sql](src/data/schema.sql) – Canonical Supabase schema.

## Need Help?

Troubleshooting tips for Supabase auth, OpenRouter/OpenAI quotas, low match quality, and general FAQs live in [docs/SETUP.md](docs/SETUP.md).

# Project Overview

AI-powered content categorization system for WordPress VIP that uses a **cascading multi-stage workflow** to map taxonomy pages to WordPress content. The system ingests content from WordPress REST API, stores it in Supabase, and uses OpenAI-compatible APIs (OpenRouter) for embeddings-based semantic matching with LLM fallback.

**Key workflow:** Load taxonomy → Ingest WordPress content → Cascading matching (semantic 0.85 → LLM 0.9 → human review) → Export results

## Documentation Index

Use progressive disclosure - only review the below indexed documents when your task requires it, otherwise skip them:

| Task | Reference |
|------|-----------|
| Testing patterns & fixtures | [tests/AGENTS.md](tests/AGENTS.md) |
| Database schema | [src/data/schema.sql](src/data/schema.sql) |
| DSPy optimization quick start | [docs/OPTIMIZATION_QUICKSTART.md](docs/OPTIMIZATION_QUICKSTART.md) |
| DSPy/GEPA best practices | [docs/DSPY_GEPA_BEST_PRACTICES.md](docs/DSPY_GEPA_BEST_PRACTICES.md) |
| Project DSPy implementation | [docs/DSPY_IMPLEMENTATION.md](docs/DSPY_IMPLEMENTATION.md) |
| Environment setup | [docs/SETUP.md](docs/SETUP.md) |
| Quick start | [docs/QUICKSTART.md](docs/QUICKSTART.md) |

## Project Structure (High Level)

- `src/` – Core application code (CLI, config, models, services, connectors, exporters, DSPy optimization).
- `tests/` – Unit and integration tests with shared fixtures.
- `data/` – Example taxonomy CSV and other input/output artifacts.
- Root files – `pyproject.toml` and related tooling scripts.

## Build & Test

- Install: `pip install -e ".[dev]"`
- Verify setup: `python scripts/test_setup.py`
- Initialize database: run `src/data/schema.sql` in Supabase (or use `python -m src.cli init-db`), then run the CLI `full-run` command with an output file (e.g., `python -m src.cli full-run --output results/results.csv`)
- Run full quality gate: `make quality-check`
- Run tests only: `pytest`
- Run pre-commit checks: `pre-commit run --all-files` (mirrors the git hook)
- CLI help and commands: `python -m src.cli --help`
- Managed runs & resumable matching: `python -m src.cli workflow start|resume|status`

## Architecture Overview

- **Ingestion** (`IngestionService`) fetches WordPress REST API content and stores it in Supabase.
- **Matching workflow** (`WorkflowService`) runs cascading semantic → LLM → human-review matching via `MatchingService` and `CategorizationService`.
- **Export** (`CSVExporter`) reads from the `export_results` view to produce analyst-friendly CSVs.

Key modules:

- `src/config.py` – Pydantic `Settings` and workflow thresholds/toggles.
- `src/models.py` – Pydantic v2 models, including `MatchStage`.
- `src/data/supabase_client.py` – Typed Supabase CRUD wrapper.
- `src/connectors/wordpress_vip.py` – WordPress VIP `/wp-json/wp/v2/posts` connector.
- `src/services/workflow.py` / `matching.py` / `categorization.py` – Workflow orchestration, semantic matching, and LLM categorization.

Database highlights (see `src/data/schema.sql` for details):

- `wordpress_content` – Ingested posts/pages and metadata.
- `taxonomy_pages` – Source taxonomy URLs and descriptors.
- `categorization_results` – LLM categorization metadata and confidence.
- `matching_results` – Canonical matches and stage metadata.
- `export_results` – View joining taxonomy, content, and match data.

## Configuration

All configuration is via `.env` (see `.env.example`):

- Required: `SUPABASE_URL`, `SUPABASE_KEY`, `WORDPRESS_VIP_SITES`
- Semantic embeddings: `SEMANTIC_API_KEY`, `SEMANTIC_BASE_URL`, `SEMANTIC_EMBEDDING_MODEL`
- LLM categorization: `LLM_API_KEY`, `LLM_BASE_URL`, `LLM_MODEL`, `LLM_BATCH_TIMEOUT`
- Matching workflow:
  - `SIMILARITY_THRESHOLD` (default 0.85)
  - `LLM_CONFIDENCE_THRESHOLD` (default 0.9)
  - `ENABLE_SEMANTIC_MATCHING`, `ENABLE_LLM_CATEGORIZATION`
- Taxonomy file: `TAXONOMY_FILE_PATH` (default `./data/taxonomy.csv`)

`OPENAI_*` variables exist for backward compatibility but should be replaced with the semantic/LLM settings above.

## Testing & Quality Gates

- Target: ≥80% coverage (enforced by pytest).
- Unit tests in `tests/unit/` mock external dependencies.
- Integration tests in `tests/integration/` are marked `@pytest.mark.integration` and are **not** included in the default `make quality-check` gate.
- Slow tests should be marked `@pytest.mark.slow` and run explicitly, not as part of automated gates.
- `make quality-check` runs Black (check), Ruff, MyPy (strict on `src`), and pytest with coverage over **non-integration, non-slow** tests.

## Critical Requirements - Never Violate

### Quality Gate Compliance

All commits must pass: `black --check`, `ruff check`, `mypy src`, `pytest --cov-fail-under=80`
Never bypass, suppress, or alter quality gates
Never use `--skip`, `--no-verify`, or disable checks
Never comment out or conditionally disable failing code
Fix root cause - do not work around quality gates

### No Placeholders in Production

Never use placeholders, TODOs, or stubs outside test contexts
Never use hardcoded values, mock data, or temporary workarounds in production
All code must be complete and production-ready

### Security Requirements

Never commit secrets, API keys, or credentials
Never expose sensitive info in logs or database
Never use string concatenation for SQL - always parameterized queries
Load all env vars from `.env` via `pydantic-settings`

## Development Practices

- Keep imports grouped and alphabetized: stdlib → third-party → local.
- Remove unused imports/variables (e.g., via `ruff check --fix`).
- Prefer modern `isinstance(value, int | float)` syntax.
- Only mock attributes that exist; mock upstream clients like `openai.OpenAI` or HTTP layers rather than internal details.
- For focused debugging runs, you can disable coverage (e.g., `pytest --no-cov tests/unit/test_x.py`).

### CLI & Service Patterns

- New CLI commands live in `src/cli.py`, use `@cli.command()` and `@click.option`, and should construct `Settings` → `SupabaseClient` → service instances.
- New services in `src/services/`:
  - Accept `Settings` and `SupabaseClient` in `__init__`.
  - Use `tenacity` `@retry` for external calls.
  - Are fully type hinted and use module-level loggers.

### Pydantic Models

- All models in `src/models.py` use Pydantic v2.
- Use `HttpUrl` for URLs and `Field()` for defaults/validation.
- Models should serialize cleanly for Supabase JSON storage via `model_config`.

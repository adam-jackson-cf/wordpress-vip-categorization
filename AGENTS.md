# CLAUDE.md

## Project Overview

AI-powered content categorization system for WordPress VIP that uses a **cascading multi-stage workflow** to map taxonomy pages to WordPress content. The system ingests content from WordPress REST API, stores it in Supabase, and uses OpenAI-compatible APIs (OpenRouter) for embeddings-based semantic matching with LLM fallback.

**Key workflow:** Load taxonomy → Ingest WordPress content → Cascading matching (semantic 0.85 → LLM 0.9 → human review) → Export results

**Cascading stages:**
1. Semantic matching (threshold: 0.85) - embedding-based similarity
2. LLM categorization (threshold: 0.9) - fallback for items below semantic threshold
3. Human review - items failing both stages exported with blank target URLs


## Project Structure

```
wordpress-vip-categorization/
├── src/
│   ├── __init__.py
│   ├── cli.py                    # Command-line interface
│   ├── config.py                 # Configuration management
│   ├── models.py                 # Pydantic data models
│   ├── data/
│   │   ├── __init__.py
│   │   └── supabase_client.py   # Supabase integration
│   ├── connectors/
│   │   ├── __init__.py
│   │   └── wordpress_vip.py     # WordPress VIP API connector
│   ├── services/
│   │   ├── __init__.py
│   │   ├── categorization.py    # OpenAI Batch categorization
│   │   ├── matching.py          # Semantic matching
│   │   └── ingestion.py         # Content ingestion orchestration
│   ├── optimization/
│   │   ├── __init__.py
│   │   ├── dspy_optimizer.py    # DSPy prompt optimization
│   │   └── evaluator.py         # Evaluation framework
│   └── exporters/
│       ├── __init__.py
│       └── csv_exporter.py      # CSV export functionality
├── tests/
│   ├── unit/
│   │   ├── test_wordpress_vip.py
│   │   ├── test_categorization.py
│   │   ├── test_matching.py
│   │   └── test_supabase_client.py
│   ├── integration/
│   │   ├── test_full_pipeline.py
│   │   └── test_batch_processing.py
│   └── conftest.py              # Pytest fixtures
├── data/
│   └── taxonomy.csv.example     # Example taxonomy file
├── pyproject.toml               # Project configuration
├── README.md                    # This file
└── .env.example                 # Environment variables template
```

## Development Commands

### Setup & Installation

```bash
# Install dependencies
pip install -e ".[dev]"

# Verify setup (checks Supabase, OpenRouter, WordPress connections)
python test_setup.py

# One-time database initialization: Run schema.sql in Supabase SQL Editor
# Then run automated setup:
python run_setup.py
```

### Testing

```bash
# Run entire suite (54% coverage gate enforced via pytest.ini)
pytest

# Generate an HTML coverage report
pytest --cov=src --cov-report=html

# Focus on units vs. integration
pytest tests/unit/ -v
pytest tests/integration/ -v -m integration

# Exercise a single test file (example)
pytest tests/test_categorization.py
```

### Code Quality

```bash
# Run all quality checks (must pass before commits)
make quality-check

# Individual checks
black src tests           # Format code
black --check src tests   # Check formatting without changes
ruff check src tests      # Lint code
mypy src                  # Type check (strict mode)
```

### CLI Commands

```bash
# Load taxonomy from CSV
python -m src.cli load-taxonomy

# Ingest WordPress content
python -m src.cli ingest
python -m src.cli ingest --sites https://site1.com,https://site2.com
python -m src.cli ingest --max-pages 10

# Run cascading matching workflow (semantic → LLM fallback)
python -m src.cli match                    # Full workflow (both stages)
python -m src.cli match --threshold 0.80   # Custom semantic threshold
python -m src.cli match --skip-llm         # Semantic matching only
python -m src.cli match --skip-semantic    # LLM categorization only
python -m src.cli match --no-batch         # Non-batch embedding mode

# Export results to CSV
python -m src.cli export --output results.csv

# View statistics
python -m src.cli stats

# Evaluate matching quality
python -m src.cli evaluate
```

## Architecture

### Core Data Flow

1. **Ingestion** (`IngestionService`): Fetches content from WordPress REST API → stores in Supabase
2. **Cascading Matching** (`WorkflowService`): Orchestrates multi-stage matching workflow
   - Stage 1: Semantic matching (0.85 threshold) via `MatchingService`
   - Stage 2: LLM categorization (0.9 threshold) via `CategorizationService` for unmatched items
   - Stage 3: Mark remaining items as needs human review
3. **Export** (`CSVExporter`): Queries database → exports matches with stage metadata to CSV

### Key Components

- **`src/config.py`**: Pydantic Settings for environment configuration. Uses `.env` file. All settings validated at startup. Includes stage toggles and thresholds.
- **`src/models.py`**: Pydantic models for all data structures. Includes `MatchStage` enum for tracking workflow stages.
- **`src/data/supabase_client.py`**: Database client wrapping Supabase SDK. Handles all CRUD operations with type-safe models.
- **`src/connectors/wordpress_vip.py`**: WordPress REST API connector. Uses `/wp-json/wp/v2/posts` endpoint with pagination.
- **`src/services/workflow.py`**: Orchestrates cascading semantic → LLM workflow with configurable stages
- **`src/services/matching.py`**: Semantic matching using embeddings. Includes `get_unmatched_taxonomy()` for fallback stage.
- **`src/services/categorization.py`**: LLM categorization with confidence thresholding. Includes `categorize_for_matching()` for workflow integration.
- **`src/cli.py`**: Click-based CLI with all user-facing commands

### Database Schema (Supabase)

- **`wordpress_content`** – Ingested WordPress posts/pages plus metadata used during matching.
  - Fields: `id` (uuid PK), `url` (unique text), `title`, `content`, `site_url`, `published_date`, `metadata` (jsonb for arbitrary post data), `created_at`.

- **`taxonomy_pages`** – Source taxonomy entries that must be matched to WordPress content.
  - Fields: `id` (uuid PK), `url` (unique text), `category`, `description`, `keywords` (if present in CSV), `created_at`.

- **`categorization_results`** – Stores AI categorization metadata, primarily when the OpenAI/OpenRouter fallback workflows are run asynchronously.
  - Fields: `id` (uuid PK), `content_id` (uuid FK → `wordpress_content`), `category`, `confidence` (float), `batch_id` (text identifier if batches are used), `created_at`.

- **`matching_results`** – Canonical junction table holding semantic similarity scores and final match metadata for the cascading workflow.
  - Fields: `id` (uuid PK), `taxonomy_id` (uuid FK → `taxonomy_pages`), `content_id` (uuid FK → `wordpress_content`, nullable when items need review), `similarity_score` (float), `match_stage` (`semantic_matched`, `llm_categorized`, `needs_human_review`), `failed_at_stage`, `created_at`.

- **`export_results`** (view) – Pre-joined projection of taxonomy + content + match metadata for reporting/CSV export. Columns include `source_url`, `target_url`, `category`, `similarity_score`, `confidence`, and stage metadata, ensuring downstream analysts never query raw tables directly.

### OpenRouter Integration

The project uses OpenRouter (OpenAI-compatible API) for cost-effective embeddings:

- **Chat model**: `google/gemini-2.0-flash-exp:free` (for future categorization)
- **Embedding model**: `qwen/qwen3-embedding-0.6b` (1024-dimensional vectors)
- **Base URL**: `https://openrouter.ai/api/v1`

**Important:** OpenRouter does NOT support OpenAI Batch API for batch categorization. However, the LLM fallback stage uses standard chat completions API which works with OpenRouter.

### Cascading Matching Workflow

**Stage 1: Semantic Matching** (threshold: 0.85)
1. Create text representations:
   - Taxonomy: `Category: {cat}\nDescription: {desc}\nKeywords: {kw1, kw2, ...}`
   - Content: `Title: {title}\n\nContent: {first_1000_chars}`
2. Generate embeddings via OpenRouter API (batch mode for efficiency)
3. Compute cosine similarity between taxonomy and content embeddings
4. Items ≥ 0.85 → store with `match_stage=semantic_matched`
5. Items < 0.85 → proceed to Stage 2

**Stage 2: LLM Categorization** (threshold: 0.9)
1. For each unmatched taxonomy page, LLM evaluates all content items
2. LLM returns best match index and confidence score
3. Items with confidence ≥ 0.9 → store with `match_stage=llm_categorized`
4. Items < 0.9 → proceed to Stage 3

**Stage 3: Human Review**
1. Items failing both stages → store with `match_stage=needs_human_review`, `content_id=NULL`
2. Export to CSV with blank `target_url` for manual review

## Configuration

All configuration via `.env` file (see `.env.example`):

- **Required**: `SUPABASE_URL`, `SUPABASE_KEY`, `WORDPRESS_VIP_SITES`
- **Semantic embeddings**: `SEMANTIC_API_KEY`, `SEMANTIC_BASE_URL`, `SEMANTIC_EMBEDDING_MODEL`
- **LLM categorization**: `LLM_API_KEY`, `LLM_BASE_URL`, `LLM_MODEL`, `LLM_BATCH_TIMEOUT`
- **Matching Workflow**:
  - `SIMILARITY_THRESHOLD` (0-1, default: 0.85) - Semantic matching threshold
  - `LLM_CONFIDENCE_THRESHOLD` (0-1, default: 0.9) - LLM categorization threshold
  - `ENABLE_SEMANTIC_MATCHING` (true/false, default: true) - Toggle semantic stage
  - `ENABLE_LLM_CATEGORIZATION` (true/false, default: true) - Toggle LLM fallback stage
- **Taxonomy**: `TAXONOMY_FILE_PATH` (default: `./data/taxonomy.csv`)

`OPENAI_*` variables remain as fallbacks for backward compatibility but should be phased out in favor of the dedicated semantic/LLM settings.

## Testing Strategy

- **Unit tests** (`tests/unit/`): Mock external APIs (Supabase, OpenRouter, WordPress). Test individual services in isolation.
- **Integration tests** (`tests/integration/`): Marked with `@pytest.mark.integration`. Test full pipeline with real dependencies.
- **Fixtures** (`tests/conftest.py`): Shared fixtures for settings, mock clients, sample data

**Coverage requirements:**
- Minimum: 54% (enforced by pytest)
- Excluded: CLI (`src/cli.py`), DSPy optimization modules (require LLM mocks)

## Quality Gates

All commits must satisfy the full gate: Black formatting, Ruff linting, MyPy strict typing, and pytest with ≥54% coverage. `make quality-check` runs the exact sequence (`black --check src tests`, `ruff check src tests`, `mypy src`, and `pytest --cov=src --cov-report=term-missing --cov-report=html`).

### Pre-commit Hooks

Install the bundled hooks to enforce formatting, linting, typing, and tests before every commit:

```bash
pre-commit install
```

You can run the entire chain on demand with:

```bash
pre-commit run --all-files
```

The hook order is: housekeeping fixes (EOF, whitespace, large files) → Black auto-format → Ruff lint/fix → `make quality-check`. Commits are blocked until each stage succeeds.

## Common Development Patterns

### Adding a new CLI command

1. Add command function in `src/cli.py` decorated with `@cli.command()`
2. Import required services
3. Initialize Settings → SupabaseClient → Service(s)
4. Use click decorators for options (`@click.option()`)
5. Add unit test in `tests/unit/test_cli.py` (if exists) or integration test

### Adding a new service

1. Create service class in `src/services/`
2. Accept `Settings` and `SupabaseClient` in `__init__`
3. Use `@retry` decorator from `tenacity` for external API calls
4. Type hint all parameters and return values
5. Add logging via `logger = logging.getLogger(__name__)`
6. Write unit tests with mocked dependencies

### Working with Pydantic models

- All models in `src/models.py` use Pydantic v2
- Use `HttpUrl` for URL validation
- Use `Field()` for defaults, validation, descriptions
- Models serialize to JSON for Supabase storage
- `model_config` sets serialization behavior

## Important Notes

- **Cascading workflow** runs semantic matching (0.85) → LLM categorization (0.9) → human review by default. Both stages configurable.
- **Stage toggles** allow semantic-only, LLM-only, or both stages. Use CLI flags (`--skip-semantic`, `--skip-llm`) or environment variables.
- **LLM fallback** works with OpenRouter (uses chat completions API). Batch categorization requires OpenAI API.
- **Database initialization** must be done manually via Supabase SQL Editor (run `schema.sql` once)
- **Service role key** required for Supabase (anon key needs RLS policies configured)
- **WordPress REST API** must be enabled on target sites (`/wp-json/wp/v2/` endpoint)
- **Match stage metadata** stored in database and exported to CSV for tracking workflow progress

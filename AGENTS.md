# Engineering Contract

This file documents the minimum structure, behaviors, and coding rules expected across the project. Use it alongside folder-level `AGENTS.md` files for specifics.

## Project Structure
- `src/` contains CLI entrypoints, connectors, services, exporters, config, and DSPy modules; keep domain logic here.
- `tests/` houses unit and integration suites with shared fixtures; mark boundary-crossing suites under `tests/integration/`.
- `data/` stores taxonomy CSVs and export artifacts; database schema lives in `src/data/schema.sql` and must be the only migration source.
- `docs/` captures detailed workflows referenced by the README; do not replicate that depth elsewhere in the tree.

## Expected Behaviors
- `make quality-check` must pass (Black, Ruff, `mypy src`, `pytest --cov=src --cov-fail-under=80`) before any commit ships.
- Unit tests mock external services; integration tests are explicitly marked `@pytest.mark.integration` and are excluded from the default gate but must be green before release.
- Slow paths belong behind `@pytest.mark.slow` and must not run in CI unless explicitly requested.
- Database changes ship through `src/data/schema.sql` + `python -m src.cli init-db`; never apply ad-hoc SQL elsewhere.
- No secrets, placeholders, or mock/test data live in production code or configuration.
- Modules excluded from coverage must be listed under `[tool.coverage.run].omit` with a short justification in the PR description; don’t silently relax the gate.
- Deterministic orchestration logic (rubric gating, demotion, workflow stats, etc.) must have targeted unit tests even if the surrounding module is exempted from coverage.
- `.cursor/plans/**` are human-authored plans; do not auto-format or reflow them unless the plan content is actually changing.

## Testing & Coverage

- The unit suite is scoped to modules not listed in `pyproject.toml`’s coverage omit list. If you introduce a new helper/service, either write tests or update the omit list with a rationale.
- When adding helper wrappers for network/storage calls (e.g., `_post_schema_sql_with_retry`), expose them at module scope and have tests patch the helper instead of the third-party client.
- Integration-only flows (CLI, connectors, Supabase client, batch orchestration) should document their manual/integration test strategy before being excluded from coverage.

## Coding Rules

### Imports & Types
- Order imports as stdlib → third-party → local, alphabetical within each block, and annotate every function signature with modern union syntax.

```python
from collections.abc import Sequence
from typing import Any

from pydantic import BaseModel

from src.models import TaxonomyPage


def score_candidates(candidates: Sequence[TaxonomyPage]) -> list[float]:
    return [candidate.score for candidate in candidates]
```

### Configuration & Models
- Load configuration exclusively via `pydantic_settings`; never pull `os.environ` directly, and keep Pydantic models v2-native (`Field`, `HttpUrl`, `model_config`).

```python
from pydantic import BaseModel, Field, HttpUrl
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    supabase_url: str
    supabase_key: str
    similarity_threshold: float = 0.85
    llm_rubric_topic_min: float = 0.8


class TaxonomyPage(BaseModel):
    url: HttpUrl
    category: str
    keywords: list[str] = Field(default_factory=list)
```

### External I/O & Logging
- Wrap every outbound network or storage call with `tenacity` retries, log via `logging.getLogger(__name__)`, and re-raise on failure; never log secrets or swallow exceptions.

```python
import logging
from tenacity import retry, stop_after_attempt, wait_exponential

logger = logging.getLogger(__name__)


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
async def fetch_post(post_id: str) -> dict[str, object]:
    try:
        return await client.get_post(post_id)
    except Exception as exc:  # pragma: no cover
        logger.error("Failed to fetch %s", post_id, exc_info=True)
        raise
```

### Services, CLI, and Dependency Injection
- CLI commands must instantiate `Settings` → `SupabaseClient` → services, and services only accept dependencies through their constructors; no hidden globals or singletons.

```python
@click.command()
def full_run() -> None:
    settings = Settings()
    client = SupabaseClient(settings=settings)
    workflow = WorkflowService(settings=settings, client=client)
    asyncio.run(workflow.full_run())
```

### Available CLI Commands - use these for running semantic, llm and other categorisation processes

- All Supabase access flows through `SupabaseClient` with parameterized queries; never concatenate SQL, and never store secrets or TODO placeholders in source.

- **Operational Flow** – If your network requires a corp certificate bundle, run each CLI via `scripts/corp_ca_exec.sh` (export env vars with `source .env` first) so OpenAI and Supabase calls succeed.
    1. Load taxonomy: `uv run python -m src.cli load-taxonomy --taxonomy-file <csv>`
    2. Ingest WordPress content: `uv run python -m src.cli ingest --sites <comma-sites> --max-pages N`
    3. Run matching: `uv run python -m src.cli match --threshold T` (run inside tmux for long jobs)
    4. Export snapshot CSV: `uv run python -m src.cli export --output results/<file>.csv`
    5. DSPy/GEPA optimization: `uv run python -m src.cli optimize-dataset --dataset data/dspy_training_dataset.csv --optimizer gepa --budget medium`, then promote with `uv run python prompt-optimiser/scripts/promote_optimized_model.py`

## Development Helpers

### When I need to analyse data quality

Run `scripts/data_quality_orchestrate.sh` to sequence analysis tasks, they can also be run directly:
- Evaluate semantic similarity thresholds (`scripts/helpers/evaluate_semantic_thresholds.py`)
- Analyze missing detection signals (`scripts/helpers/analyze_missing_signals.py`)
- Extract sample matches for review (`scripts/helpers/extract_sample_matches.py`)
- Generate comprehensive analysis report (`scripts/helpers/generate_report.py`)

### When I need to set up or reset infrastructure

Run `scripts/setup_orchestrate.sh` to sequence setup tasks:
- Test setup and configuration (`scripts/helpers/test_setup.py`)
- Initialize/reset Supabase schema (`scripts/helpers/reset_supabase_local.py`)

### When I need to run DSPy/GEPA optimization

Run `prompt-optimiser/scripts/orchestrate.sh` to sequence the optimization workflow, they can also be run:
- Generate training dataset (`prompt-optimiser/scripts/generate_dspy_dataset.py` or `prompt-optimiser/scripts/build_dspy_dataset.py`)
- Run bootstrap smoke test (`prompt-optimiser/scripts/run_quick_optimization_test.py`)
- Run full GEPA optimization (`prompt-optimiser/scripts/run_optimization_workflow.py`)
- Promote optimized model (`prompt-optimiser/scripts/promote_optimized_model.py`)

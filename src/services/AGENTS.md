# Service Layer Rules

Applies to orchestration/services inside `src/services/` (workflow, ingestion, matching, categorization).

## Construction & Dependencies
- Services accept `Settings` plus their collaborators (e.g., `SupabaseClient`, connectors) via `__init__`; never instantiate dependencies inside methods.
- Keep each service single-responsibility (ingestion vs. matching vs. exporting) and avoid mixing orchestration with data access.

```python
class MatchingService:
    def __init__(self, *, settings: Settings, client: SupabaseClient) -> None:
        self.settings = settings
        self.client = client
```

## Logging & Error Handling
- Use `logging.getLogger(__name__)` once per module, log structured context (ids, URLs), and never leak secrets.
- Wrap outbound calls in try/except, re-raise after logging, and rely on `tenacity` decorators for retries.

```python
logger = logging.getLogger(__name__)


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
async def _fetch_candidates(self, taxonomy_id: str) -> list[MatchResult]:
    try:
        return await self.client.fetch_candidates(taxonomy_id)
    except Exception as exc:
        logger.error("Candidate fetch failed for %s", taxonomy_id, exc_info=True)
        raise
```

## Workflow Composition
- Workflow services orchestrate semantic → LLM → human-review stages sequentially and must honor the configured thresholds (0.85 semantic, 0.9 LLM unless overridden).
- Surface stage metadata (`MatchStage`) back into Supabase for downstream exports; never skip audit fields like confidence or stage.

```python
async def run_stage(self, taxonomy_id: str) -> MatchResult | None:
    semantic = await self.semantic.match(taxonomy_id)
    if semantic and semantic.similarity >= self.settings.similarity_threshold:
        return semantic
    return await self.categorizer.categorize(taxonomy_id)
```

## Testing Expectations
- Unit tests mock every dependency (Settings + SupabaseClient + outbound connectors) and assert threshold behavior.
- Integration tests may touch Supabase or WordPress sandboxes but must live in `tests/integration/` and be marker-gated.

## Checklist (`src/services/`)
- Constructor injects all dependencies; no hidden globals/singletons.
- Logging uses module-level logger with contextual metadata and zero secrets.
- External calls include `tenacity` retries plus try/except + re-raise wrappers.
- Thresholds, match stages, and audit metadata honor `Settings` values.
- Unit tests cover success/error paths with mocked dependencies.

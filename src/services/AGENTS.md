# Service Layer Development Standards

Project overview: [../../AGENTS.md](../../AGENTS.md)
Code review checklist: [../../CODE_REVIEW.md](../../CODE_REVIEW.md)

## Service Layer Pattern

```python
import logging
from src.config import Settings
from src.data.supabase_client import SupabaseClient

logger = logging.getLogger(__name__)

class MatchingService:
    def __init__(self, settings: Settings, client: SupabaseClient):
        self.settings = settings
        self.client = client
        self.threshold = settings.similarity_threshold

    async def match_taxonomy(self, taxonomy_id: str) -> list[MatchResult]:
        logger.info(f"Starting matching for taxonomy {taxonomy_id}")
        # Implementation
```

Accept `Settings` and `SupabaseClient` in `__init__`
Use dependency injection - don't create dependencies inside services
Keep services focused - single responsibility
Add logging via `logging.getLogger(__name__)`

## Dependency Injection Requirements

- **Settings**: Always accept `Settings` instance in constructor
- **SupabaseClient**: Always accept `SupabaseClient` instance in constructor
- **No Direct Instantiation**: Services should never create their own dependencies
- **Constructor Injection**: All dependencies passed in `__init__`

## Service Error Handling

```python
from tenacity import retry, stop_after_attempt, wait_exponential

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
async def call_external_api(self, url: str) -> dict[str, Any]:
    try:
        response = await self.client.get(url)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        logger.error(f"API call failed: {e}", exc_info=True)
        raise
```

Use `tenacity` for retry logic on external APIs
Always wrap async operations in try-catch
Log errors with context (no secrets)
Never silently swallow exceptions

## Service Testing

Services are unit tested with mocked dependencies:

```python
def test_matching_service_returns_high_similarity_matches(mock_supabase):
    # Arrange
    settings = Settings(similarity_threshold=0.85)
    service = MatchingService(settings, mock_supabase)

    # Act
    results = service.match_taxonomy(taxonomy_id="test-123")

    # Assert
    assert len(results) > 0
    assert all(r.similarity_score >= 0.85 for r in results)
```

See [../../tests/AGENTS.md](../../tests/AGENTS.md) for testing patterns.

## Common Service Patterns

### Workflow Orchestration
`WorkflowService` coordinates multi-stage workflows (semantic → LLM → human review)

### Data Processing
Services handle single data transformation or business logic operation

### External Integration
Services wrap external API calls (WordPress, OpenRouter, OpenAI)

## Common Pitfalls

1. **Creating Dependencies**: Services should accept, not create dependencies
2. **Missing Error Handling**: Always wrap async operations in try-catch
3. **Multiple Responsibilities**: Keep services focused on single responsibility
4. **Missing Logging**: Use `logging.getLogger(__name__)` for all services

## Service Checklist (src/services)

- [ ] Service `__init__` accepts `Settings` and `SupabaseClient` (or other dependencies) via constructor
- [ ] No direct instantiation of external clients or settings inside service methods
- [ ] Service has a clear, single responsibility
- [ ] Logging uses `logging.getLogger(__name__)` with useful context (no secrets)
- [ ] External API calls use `tenacity` retries and proper error handling
- [ ] Service logic is covered by unit tests with mocked dependencies

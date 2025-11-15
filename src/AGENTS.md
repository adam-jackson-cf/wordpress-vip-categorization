# Python & Pydantic Development Standards

Project overview: [../AGENTS.md](../AGENTS.md)
Code review checklist: [../CODE_REVIEW.md](../CODE_REVIEW.md)

## Import Organization

```python
from datetime import datetime
from typing import Any, Optional

import pytest
from pydantic import BaseModel

from src.models import TaxonomyPage
from src.config import Settings
```

Organize imports: stdlib → third-party → local (alphabetical within each group)

## Type Hints

Use explicit type hints for all parameters and returns
Modern isinstance: `isinstance(value, int | float)`
Avoid `Any` - use proper types or `Unknown`
Must pass `mypy src` strict mode

## Error Handling Pattern

```python
from tenacity import retry, stop_after_attempt, wait_exponential
import logging

logger = logging.getLogger(__name__)

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
async def fetch_data(url: str) -> dict[str, Any]:
    try:
        response = await client.get(url)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        logger.error(f"Failed to fetch {url}: {e}", exc_info=True)
        raise
```

Wrap all async operations in try-catch
Never silently swallow errors
Use `tenacity` for retry logic on external APIs
Log with context: user, request ID, endpoint (no secrets)

## Code Style

Line length: 100 chars (black enforced)
Use async/await for all I/O
Clear variable names, no magic numbers
Remove unused imports/vars (ruff F401, F841)

## Pydantic Settings Management

```python
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    supabase_url: str
    supabase_key: str
    openai_api_key: str
    similarity_threshold: float = 0.85
    llm_confidence_threshold: float = 0.9
```

Use `pydantic-settings` for all configuration
Load from `.env` files - never hardcode
Validate at startup - fail fast if invalid

## Pydantic Data Model Pattern

```python
from pydantic import BaseModel, Field, HttpUrl
from datetime import datetime
from enum import Enum

class MatchStage(str, Enum):
    SEMANTIC_MATCHED = "semantic_matched"
    LLM_CATEGORIZED = "llm_categorized"
    NEEDS_HUMAN_REVIEW = "needs_human_review"

class TaxonomyPage(BaseModel):
    id: str
    url: HttpUrl
    category: str
    description: str
    keywords: list[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.utcnow)
```

Use Pydantic v2 for all data models
Use `HttpUrl` for URL validation
Use `Field()` for defaults, validation, descriptions
Use `model_config` for serialization behavior

## Common Pitfalls to Avoid

1. **Hardcoding Configuration**: Always use Pydantic Settings with `.env` files
2. **Missing Type Hints**: All functions must have explicit type annotations
3. **Poor Error Handling**: Never silently swallow exceptions
4. **Unused Imports**: Run `ruff check --fix` to auto-remove
5. **Old isinstance Syntax**: Use `int | float` not `(int, float)`
6. **Incomplete Tests**: Ensure ≥80% coverage, mock external dependencies
7. **Logging Secrets**: Never log API keys, passwords, or sensitive data

## Per-File Checklist (src)

- [ ] Imports organized: stdlib → third-party → local (alphabetical)
- [ ] All functions have explicit parameter and return type hints
- [ ] Modern `isinstance` syntax used (e.g., `int | float`)
- [ ] No unused imports or variables (passes Ruff F401/F841)
- [ ] Configuration loaded via Pydantic Settings (no hardcoded env values)
- [ ] Pydantic models follow v2 patterns (`Field`, `HttpUrl`, `model_config`)
- [ ] Async I/O uses `async/await` with proper error handling and logging

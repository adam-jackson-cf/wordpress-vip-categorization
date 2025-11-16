# src Coding Rules

Covers Python modules inside `src/` (CLI, config, services, connectors, exporters, DSPy modules).

## Imports & Style
- Order imports stdlib → third-party → local, alphabetized per block, and keep line length ≤100 enforced by Black.
- Prefer descriptive names and remove unused imports/variables so Ruff stays clean without `noqa` escapes.
- Use async/await for every I/O path; never block the event loop inside services or connectors.

```python
from datetime import datetime
from typing import Any

import httpx

from src.config import Settings
from src.models import MatchResult
```

## Type Hints & Pydantic
- Annotate every parameter/return with concrete types and modern union syntax (`int | float`).
- Keep Pydantic models v2-native with `Field`, `HttpUrl`, and `model_config`; model instances must serialize cleanly for Supabase JSON columns.

```python
from pydantic import BaseModel, Field, HttpUrl


class MatchResult(BaseModel):
    taxonomy_id: str
    content_url: HttpUrl
    similarity: float = Field(ge=0, le=1)
```

## Settings Management
- Centralize configuration in `src/config.py` using `pydantic_settings`; never access `os.environ` elsewhere and always provide defaults or Field-level validation.

```python
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    llm_model: str
    similarity_threshold: float = 0.85
```

## Error Handling & Logging
- Guard every external call (HTTP, Supabase, OpenRouter) with try/except + contextual logging, re-raising after logging, and use `tenacity` retries for transient errors.

```python
import logging
from tenacity import retry, stop_after_attempt, wait_exponential

logger = logging.getLogger(__name__)


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=8))
async def fetch_posts(connector: WordPressConnector, site: str) -> list[dict[str, Any]]:
    try:
        return await connector.fetch(site)
    except Exception as exc:
        logger.error("WordPress fetch failed for %s", site, exc_info=True)
        raise
```

## Checklist (`src/`)
- Imports grouped + alphabetized (stdlib → third-party → local) with ≤100 character lines.
- All functions and methods have explicit type hints and avoid `Any` unless unavoidable.
- Configuration flows through `Settings` only; no direct `os.environ` access.
- External I/O uses async/await, contextual logging, and `tenacity` retries.
- Pydantic models and DTOs are v2-compliant and serialization-safe for Supabase.

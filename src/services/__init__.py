"""Services for content processing."""

from typing import Any

# Keep exports lightweight to avoid circular imports at module import time.
__all__ = [
    "CategorizationService",
    "MatchingService",
    "IngestionService",
]


def __getattr__(name: str) -> Any:  # pragma: no cover - module-level lazy loading
    if name == "CategorizationService":
        from src.services.categorization import CategorizationService

        return CategorizationService
    if name == "MatchingService":
        from src.services.matching import MatchingService

        return MatchingService
    if name == "IngestionService":
        from src.services.ingestion import IngestionService

        return IngestionService
    raise AttributeError(name)

"""Shared embedding generation helper."""

import logging
from collections.abc import Iterable

import openai
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from src.config import Settings

logger = logging.getLogger(__name__)


OPENAI_RETRY_EXCEPTIONS = (
    openai.APIError,
    openai.APIConnectionError,
    openai.APITimeoutError,
    openai.RateLimitError,
    openai.InternalServerError,
    openai.APIStatusError,
)


class EmbeddingService:
    """Wrapper around the semantic embedding provider."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.client = openai.OpenAI(
            api_key=settings.semantic_api_key,
            base_url=settings.semantic_base_url,
        )
        self.model = settings.semantic_embedding_model
        logger.info("Initialized embedding service with model %s", self.model)

    @retry(
        retry=retry_if_exception_type(OPENAI_RETRY_EXCEPTIONS),
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        reraise=True,
    )
    def embed(self, text: str) -> list[float]:
        """Create a single embedding for the provided text."""

        response = self.client.embeddings.create(
            model=self.model,
            input=text,
            encoding_format="float",
        )
        return response.data[0].embedding

    def embed_batch(self, texts: Iterable[str]) -> list[list[float]]:
        """Generate embeddings for a batch of texts.

        Args:
            texts: Iterable of text values.
        """

        embeddings: list[list[float]] = []
        for text in texts:
            embeddings.append(self.embed(text))
        return embeddings

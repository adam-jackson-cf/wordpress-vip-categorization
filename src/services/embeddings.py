"""Shared embedding generation helper."""

import asyncio
import logging
from collections.abc import Sequence

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

    def embed_batch(self, texts: Sequence[str]) -> list[list[float]]:
        """Generate embeddings for a batch of texts using parallel chunk processing.

        Processes texts in chunks using OpenAI's batch API with concurrent execution
        across chunks. Preserves input order and handles failures with retry logic.

        Args:
            texts: Sequence of text values to embed.

        Returns:
            List of embeddings in the same order as input texts.

        Raises:
            Exception: If chunks fail after retries, raises with structured error info.
        """
        if not texts:
            return []

        # Convert to list for indexing and chunking
        text_list = list(texts)
        batch_size = self.settings.embedding_batch_size

        # Run async batch processing synchronously
        # Use asyncio.run() which creates a new event loop if needed
        # This works for our use case since we're always called from sync code
        try:
            # Try to get existing loop first (for compatibility)
            asyncio.get_running_loop()
            # If we're in an async context, we can't use asyncio.run()
            # This shouldn't happen in our use case, but handle gracefully
            import concurrent.futures

            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(
                    lambda: asyncio.run(self._embed_batch_async(text_list, batch_size))
                )
                return future.result()
        except RuntimeError:
            # No running loop, safe to use asyncio.run()
            return asyncio.run(self._embed_batch_async(text_list, batch_size))

    async def _embed_batch_async(self, texts: list[str], batch_size: int) -> list[list[float]]:
        """Async implementation of batch embedding with parallel chunk processing.

        Args:
            texts: List of text values to embed.
            batch_size: Maximum number of texts per chunk.

        Returns:
            List of embeddings in the same order as input texts.
        """
        if not texts:
            return []

        # Chunk texts into batches
        chunks: list[list[str]] = []
        for i in range(0, len(texts), batch_size):
            chunks.append(texts[i : i + batch_size])

        if len(chunks) == 1:
            # Single chunk: use synchronous batch API directly
            try:
                return await asyncio.to_thread(self._embed_chunk, chunks[0])
            except Exception as exc:
                # Chunk failed after retries, raise RuntimeError with structured info
                error_msg = (
                    f"Failed to generate embeddings for {len(texts)} texts "
                    f"(indices: {list(range(len(texts)))})"
                )
                logger.error(error_msg, exc_info=True)
                raise RuntimeError(error_msg) from exc

        # Multiple chunks: process in parallel
        chunk_tasks = [asyncio.to_thread(self._embed_chunk, chunk) for chunk in chunks]
        chunk_results = await asyncio.gather(*chunk_tasks, return_exceptions=True)

        # Reassemble results in original order, handling failures
        all_embeddings: list[list[float] | None] = [None] * len(texts)
        failed_indices: list[int] = []

        for chunk_idx, result in enumerate(chunk_results):
            chunk_start = chunk_idx * batch_size
            chunk_end = min(chunk_start + batch_size, len(texts))

            if isinstance(result, BaseException):
                # Chunk failed after retries
                failed_indices.extend(range(chunk_start, chunk_end))
                logger.error(
                    "Chunk %d (indices %d-%d) failed after retries: %s",
                    chunk_idx,
                    chunk_start,
                    chunk_end - 1,
                    result,
                    exc_info=True,
                )
            else:
                # Place embeddings back into their original positions
                chunk_embeddings = result
                expected_len = chunk_end - chunk_start
                if len(chunk_embeddings) != expected_len:
                    logger.warning(
                        "Mismatch between chunk size and embeddings returned "
                        "(expected %d, got %d); truncating to expected length",
                        expected_len,
                        len(chunk_embeddings),
                    )
                    if len(chunk_embeddings) < expected_len:
                        missing_indices = list(
                            range(chunk_start + len(chunk_embeddings), chunk_end)
                        )
                        failed_indices.extend(missing_indices)
                    chunk_embeddings = chunk_embeddings[:expected_len]

                for offset, embedding in enumerate(chunk_embeddings):
                    all_embeddings[chunk_start + offset] = embedding

        if failed_indices:
            # Find the first exception to use as the cause
            first_exception: Exception | None = None
            for result in chunk_results:
                if isinstance(result, Exception):
                    first_exception = result
                    break

            error_msg = (
                f"Failed to generate embeddings for {len(failed_indices)} texts "
                f"(indices: {failed_indices[:10]}{'...' if len(failed_indices) > 10 else ''})"
            )
            logger.error(error_msg)
            if first_exception:
                raise RuntimeError(error_msg) from first_exception
            raise RuntimeError(error_msg)

        # Safe to assert no None values remain
        return [embedding for embedding in all_embeddings if embedding is not None]

    @retry(
        retry=retry_if_exception_type(OPENAI_RETRY_EXCEPTIONS),
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        reraise=True,
    )
    def _embed_chunk(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for a single chunk of texts using OpenAI batch API.

        Args:
            texts: List of text values in this chunk.

        Returns:
            List of embeddings in the same order as input texts.
        """
        response = self.client.embeddings.create(
            model=self.model,
            input=texts,
            encoding_format="float",
        )
        # Response data is already in order matching input
        return [item.embedding for item in response.data]

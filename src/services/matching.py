"""Semantic matching service for taxonomy to content mapping."""

import logging
from datetime import datetime
from uuid import UUID

import numpy as np

from src.config import Settings
from src.data.supabase_client import SupabaseClient
from src.models import MatchingResult, MatchStage, TaxonomyPage, WordPressContent
from src.services.embeddings import EmbeddingService

logger = logging.getLogger(__name__)


class MatchingService:
    """Service for semantic matching between taxonomy and content.

    Uses OpenAI embeddings to compute semantic similarity between
    taxonomy pages (with keywords/descriptions) and WordPress content.
    """

    def __init__(
        self,
        settings: Settings,
        db_client: SupabaseClient,
        embedding_service: EmbeddingService | None = None,
    ) -> None:
        """Initialize matching service.

        Args:
            settings: Application settings.
            db_client: Supabase database client.
        """
        self.settings = settings
        self.db = db_client
        self.embedding_service = embedding_service or EmbeddingService(settings)
        self.embedding_model = settings.semantic_embedding_model
        logger.info(
            "Initialized matching service with model %s via %s",
            self.embedding_model,
            settings.semantic_base_url,
        )

    def get_embedding(self, text: str) -> list[float]:
        """Get embedding vector for text using the shared embedding service."""

        return self.embedding_service.embed(text)

    def create_taxonomy_text(self, taxonomy: TaxonomyPage) -> str:
        """Create text representation of taxonomy page for embedding.

        Args:
            taxonomy: Taxonomy page.

        Returns:
            Combined text for embedding.
        """
        parts = [
            f"Category: {taxonomy.category}",
            f"Description: {taxonomy.description}",
        ]

        if taxonomy.keywords:
            keywords_str = ", ".join(taxonomy.keywords)
            parts.append(f"Keywords: {keywords_str}")

        return "\n".join(parts)

    def create_content_text(self, content: WordPressContent) -> str:
        """Create text representation of content for embedding.

        Args:
            content: WordPress content.

        Returns:
            Combined text for embedding.
        """
        # Use title and first 1000 chars of content
        content_preview = content.content[:1000]
        return f"Title: {content.title}\n\nContent: {content_preview}"

    def compute_similarity(self, embedding1: list[float], embedding2: list[float]) -> float:
        """Compute cosine similarity between two embeddings.

        Args:
            embedding1: First embedding vector.
            embedding2: Second embedding vector.

        Returns:
            Cosine similarity score (0-1).
        """
        vec1 = np.array(embedding1)
        vec2 = np.array(embedding2)

        # Cosine similarity
        similarity = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

        # Convert to 0-1 range
        return float((similarity + 1) / 2)

    def _ensure_taxonomy_embedding(self, taxonomy: TaxonomyPage) -> list[float]:
        if taxonomy.taxonomy_embedding is not None:
            return taxonomy.taxonomy_embedding

        embedding = self.get_embedding(self.create_taxonomy_text(taxonomy))
        taxonomy.taxonomy_embedding = embedding
        try:
            self.db.update_taxonomy_embedding(taxonomy.id, embedding)
        except Exception as exc:  # pragma: no cover - network failure
            logger.warning("Failed to persist taxonomy embedding %s: %s", taxonomy.id, exc)
        return embedding

    def _ensure_content_embedding(self, content: WordPressContent) -> list[float]:
        if content.content_embedding is not None:
            return content.content_embedding

        embedding = self.get_embedding(self.create_content_text(content))
        content.content_embedding = embedding
        try:
            self.db.update_content_embedding(content.id, embedding)
        except Exception as exc:  # pragma: no cover - network failure
            logger.warning("Failed to persist content embedding %s: %s", content.id, exc)
        return embedding

    def _local_similarity_search(
        self,
        taxonomy: TaxonomyPage,
        taxonomy_embedding: list[float],
        limit: int,
    ) -> list[tuple[WordPressContent, float]]:
        content_items = self.db.get_all_content()
        matches: list[tuple[WordPressContent, float]] = []
        for content in content_items:
            content_embedding = self._ensure_content_embedding(content)
            similarity = self.compute_similarity(taxonomy_embedding, content_embedding)
            matches.append((content, similarity))
        matches.sort(key=lambda item: item[1], reverse=True)
        return matches[:limit]

    def match_taxonomy_to_content(
        self,
        taxonomy: TaxonomyPage,
        limit: int | None = None,
        min_threshold: float | None = None,
    ) -> list[tuple[WordPressContent, float]]:
        """Match a taxonomy page to content items.

        Args:
            taxonomy: Taxonomy page to match.
            content_items: List of content items to match against.

        Returns:
            List of (content, similarity_score) tuples, sorted by score descending.
        """
        limit = limit or self.settings.semantic_candidate_limit
        if min_threshold is None:
            min_threshold = self.settings.similarity_threshold

        taxonomy_embedding = self._ensure_taxonomy_embedding(taxonomy)

        try:
            matches = self.db.match_content_by_embedding(
                taxonomy_embedding,
                min_threshold,
                limit,
            )
        except Exception as exc:  # pragma: no cover - network failure
            logger.warning(
                "Vector RPC failed for taxonomy %s, falling back to local search: %s",
                taxonomy.id,
                exc,
            )
            matches = self._local_similarity_search(taxonomy, taxonomy_embedding, limit)

        logger.debug(
            f"Matched taxonomy {taxonomy.url} to {len(matches)} content items. "
            f"Best match score: {matches[0][1]:.3f}"
            if matches
            else "No matches found"
        )

        return matches

    def find_best_match(
        self,
        taxonomy: TaxonomyPage,
        min_threshold: float | None = None,
    ) -> tuple[WordPressContent, float] | None:
        """Find best matching content for a taxonomy page.

        Args:
            taxonomy: Taxonomy page.
            min_threshold: Minimum similarity threshold (uses settings default if None).

        Returns:
            (content, score) tuple if match found above threshold, None otherwise.
        """
        if min_threshold is None:
            min_threshold = self.settings.similarity_threshold

        matches = self.match_taxonomy_to_content(
            taxonomy, limit=self.settings.semantic_candidate_limit, min_threshold=min_threshold
        )

        if matches and matches[0][1] >= min_threshold:
            return matches[0]

        return None

    def get_unmatched_taxonomy(self, min_threshold: float | None = None) -> list[TaxonomyPage]:
        """Get taxonomy pages that are below the matching threshold.

        Args:
            min_threshold: Minimum similarity threshold (uses settings default if None).

        Returns:
            List of taxonomy pages that have no match or are below threshold.
        """
        if min_threshold is None:
            min_threshold = self.settings.similarity_threshold

        unmatched = self.db.get_unmatched_taxonomy(min_threshold)
        logger.info(
            "Found %s taxonomy pages below semantic threshold %.2f",
            len(unmatched),
            min_threshold,
        )
        return unmatched

    def match_all_taxonomy(
        self,
        taxonomy_pages: list[TaxonomyPage] | None = None,
        content_items: list[WordPressContent] | None = None,
        min_threshold: float | None = None,
        store_results: bool = True,
    ) -> dict[UUID, MatchingResult]:
        """Match all taxonomy pages to content.

        Args:
            taxonomy_pages: Optional list of taxonomy pages. If None, loads from database.
            content_items: Optional list of content items. If None, loads from database.
            min_threshold: Minimum similarity threshold.
            store_results: Whether to store results in database.

        Returns:
            Dictionary mapping taxonomy_id to best matching result (or None).
        """
        if min_threshold is None:
            min_threshold = self.settings.similarity_threshold

        # Get all taxonomy pages if not provided
        if taxonomy_pages is None:
            taxonomy_pages = self.db.get_all_taxonomy()
        logger.info(
            "Matching %s taxonomy pages via stored embeddings",
            len(taxonomy_pages),
        )

        results: dict[UUID, MatchingResult] = {}

        pending: list[MatchingResult] = []

        def flush_pending() -> None:
            if not pending or not store_results:
                return
            self.db.bulk_upsert_matchings(
                pending,
                chunk_size=self.settings.matching_batch_size,
            )
            pending.clear()

        for taxonomy in taxonomy_pages:
            best_match = self.find_best_match(taxonomy, min_threshold)

            if best_match:
                content, score = best_match
                matching_result = MatchingResult(
                    taxonomy_id=taxonomy.id,
                    content_id=content.id,
                    similarity_score=score,
                    match_stage=MatchStage.SEMANTIC_MATCHED,
                    updated_at=datetime.utcnow(),
                )

                if store_results:
                    pending.append(matching_result)
                    if len(pending) >= self.settings.matching_batch_size:
                        flush_pending()

                results[taxonomy.id] = matching_result
                logger.info(
                    f"Matched taxonomy {taxonomy.url} to {content.url} (score: {score:.3f})"
                )
            else:
                # Create result with no match
                matching_result = MatchingResult(
                    taxonomy_id=taxonomy.id,
                    content_id=None,
                    similarity_score=0.0,
                    match_stage=MatchStage.NEEDS_HUMAN_REVIEW,
                    failed_at_stage="semantic_matching",
                    updated_at=datetime.utcnow(),
                )

                if store_results:
                    pending.append(matching_result)
                    if len(pending) >= self.settings.matching_batch_size:
                        flush_pending()

                results[taxonomy.id] = matching_result
                logger.warning(
                    f"No match found for taxonomy {taxonomy.url} above threshold {min_threshold}"
                )

        flush_pending()

        matched_count = len(
            [result for result in results.values() if result.content_id is not None]
        )
        logger.info(
            f"Completed matching: {matched_count} "
            f"out of {len(taxonomy_pages)} taxonomy pages matched"
        )

        return results

    def batch_get_embeddings(self, texts: list[str]) -> list[list[float]]:
        """Get embeddings for multiple texts in batch using the shared service."""

        return self.embedding_service.embed_batch(texts)

    def match_all_taxonomy_batch(
        self,
        taxonomy_pages: list[TaxonomyPage] | None = None,
        content_items: list[WordPressContent] | None = None,
        min_threshold: float | None = None,
        store_results: bool = True,
    ) -> dict[UUID, MatchingResult]:
        """Backwards-compatible wrapper around match_all_taxonomy.

        Vector search now makes the primary flow efficient, so this method simply
        delegates to ``match_all_taxonomy`` to avoid duplicate logic.
        """

        if content_items:
            logger.info(
                "match_all_taxonomy_batch ignoring manual content pool; using stored embeddings"
            )

        return self.match_all_taxonomy(
            taxonomy_pages=taxonomy_pages,
            min_threshold=min_threshold,
            store_results=store_results,
        )

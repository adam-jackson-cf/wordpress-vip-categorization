"""Semantic matching service for taxonomy to content mapping."""

import logging
from datetime import datetime, timezone
from urllib.parse import urlparse
from uuid import UUID

import numpy as np

from src.config import Settings
from src.data.supabase_client import SupabaseClient
from src.models import MatchingResult, MatchStage, TaxonomyPage, WordPressContent
from src.services.embeddings import EmbeddingService
from src.services.language import detect_language_code

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

    @staticmethod
    def _tokenize_url_path(url_value: str) -> str:
        parsed = urlparse(url_value)
        segments = [segment for segment in parsed.path.split("/") if segment]
        if not segments:
            return "/"
        humanized = [segment.replace("-", " ").replace("_", " ") for segment in segments]
        return " > ".join(humanized)

    def _describe_audiences(self, taxonomy: TaxonomyPage) -> str:
        audiences: list[str] = []
        if taxonomy.primary_audiance:
            audiences.append(f"Primary: {taxonomy.primary_audiance}")
        if taxonomy.secondary_audiance:
            audiences.append(f"Secondary: {taxonomy.secondary_audiance}")
        return ", ".join(audiences)

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
        audiences = ", ".join(
            filter(None, [taxonomy.primary_audiance, taxonomy.secondary_audiance])
        )
        key_topics_sentence = ", ".join(taxonomy.key_topics)

        species_sentence = ", ".join(taxonomy.species)

        parts = [
            f"UID: {taxonomy.uid}" if taxonomy.uid else None,
            f"Destination Path: {self._tokenize_url_path(str(taxonomy.destination_url))}",
            f"Content Type: {taxonomy.content_type}",
            f"Audiences: {audiences}" if audiences else None,
            f"English Name: {taxonomy.english_page_name}" if taxonomy.english_page_name else None,
            f"Local Name: {taxonomy.local_page_name}" if taxonomy.local_page_name else None,
            f"Species: {species_sentence}" if species_sentence else None,
            f"Summary: {taxonomy.semantic_summary}",
            f"Key Topics: {key_topics_sentence}" if key_topics_sentence else None,
        ]

        return "\n".join(part for part in parts if part)

    def create_content_text(self, content: WordPressContent) -> str:
        """Create text representation of content for embedding.

        Args:
            content: WordPress content.

        Returns:
            Combined text for embedding.
        """
        metadata = content.metadata or {}
        slug = metadata.get("slug")
        categories = [str(value) for value in (metadata.get("categories") or [])]
        tags = [str(value) for value in (metadata.get("tags") or [])]
        excerpt = (metadata.get("excerpt") or "").strip()
        primary_excerpt = excerpt or content.content[:400]
        excerpt_line = f"Excerpt: {primary_excerpt}" if primary_excerpt else None

        preview = content.content[:2000]
        language_code = detect_language_code(preview or content.title)

        parts = [
            f"Title: {content.title}",
            f"Slug: {slug}" if slug else None,
            f"Site: {content.site_url}",
            f"URL Path: {self._tokenize_url_path(str(content.url))}",
            f"Detected Language: {language_code}",
            excerpt_line,
            f"Categories: {', '.join(categories)}" if categories else None,
            f"Tags: {', '.join(tags)}" if tags else None,
            (
                f"Published: {content.published_date.date().isoformat()}"
                if content.published_date
                else None
            ),
            f"Content Preview: {preview}",
        ]

        return "\n".join(part for part in parts if part)

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

    def _local_similarity_search_taxonomy(
        self,
        content: WordPressContent,
        content_embedding: list[float],
        limit: int,
        taxonomy_pool: dict[UUID, TaxonomyPage] | None = None,
    ) -> list[tuple[TaxonomyPage, float]]:
        if taxonomy_pool is not None:
            taxonomy_pages = list(taxonomy_pool.values())
        else:
            taxonomy_pages = self.db.get_all_taxonomy()
        matches: list[tuple[TaxonomyPage, float]] = []
        for taxonomy in taxonomy_pages:
            taxonomy_embedding = self._ensure_taxonomy_embedding(taxonomy)
            similarity = self.compute_similarity(content_embedding, taxonomy_embedding)
            matches.append((taxonomy, similarity))
        matches.sort(key=lambda item: item[1], reverse=True)
        return matches[:limit]

    def match_taxonomy_to_content(
        self,
        content: WordPressContent,
        limit: int | None = None,
        min_threshold: float | None = None,
        taxonomy_pool: dict[UUID, TaxonomyPage] | None = None,
    ) -> list[tuple[TaxonomyPage, float]]:
        """Match a content item to taxonomy pages.

        Args:
            content: Content item to match.
            limit: Maximum number of matches to return.
            min_threshold: Minimum similarity threshold.

        Returns:
            List of (taxonomy, similarity_score) tuples, sorted by score descending.
        """
        limit = limit or self.settings.semantic_candidate_limit
        if min_threshold is None:
            min_threshold = self.settings.similarity_threshold

        content_embedding = self._ensure_content_embedding(content)

        allowed_ids = set(taxonomy_pool.keys()) if taxonomy_pool else None

        try:
            matches = self.db.match_taxonomy_by_embedding(
                content_embedding,
                0.0,
                limit,
            )
        except Exception as exc:  # pragma: no cover - network failure
            logger.warning(
                "Vector RPC failed for content %s, falling back to local search: %s",
                content.id,
                exc,
            )
            matches = self._local_similarity_search_taxonomy(
                content,
                content_embedding,
                limit,
                taxonomy_pool,
            )

        if allowed_ids is not None:
            filtered = [(tax, score) for tax, score in matches if tax.id in allowed_ids]
            if filtered:
                matches = filtered
            else:
                matches = self._local_similarity_search_taxonomy(
                    content,
                    content_embedding,
                    limit,
                    taxonomy_pool,
                )

        logger.debug(
            f"Matched content {content.url} to {len(matches)} taxonomy pages. "
            f"Best match score: {matches[0][1]:.3f}"
            if matches
            else "No matches found"
        )

        return matches

    def build_candidate_map(
        self,
        content_items: list[WordPressContent],
        taxonomy_pages: list[TaxonomyPage] | None = None,
    ) -> dict[UUID, list[TaxonomyPage]]:
        """Construct a candidate taxonomy list for each content item."""

        taxonomy_lookup = {tax.id: tax for tax in taxonomy_pages} if taxonomy_pages else None
        candidate_map: dict[UUID, list[TaxonomyPage]] = {}

        for content in content_items:
            matches = self.match_taxonomy_to_content(
                content,
                limit=self.settings.llm_candidate_limit,
                min_threshold=self.settings.llm_candidate_min_score,
                taxonomy_pool=taxonomy_lookup,
            )
            filtered = [taxonomy for taxonomy, score in matches if score >= self.settings.llm_candidate_min_score]
            if not filtered:
                filtered = [taxonomy for taxonomy, _ in matches][: self.settings.llm_candidate_limit]
            if not filtered and taxonomy_pages:
                filtered = taxonomy_pages[: self.settings.llm_candidate_limit]
            candidate_map[content.id] = filtered

        return candidate_map

    def match_content_to_taxonomy(
        self,
        taxonomy: TaxonomyPage,
        limit: int | None = None,
        min_threshold: float | None = None,
    ) -> list[tuple[WordPressContent, float]]:
        """Match a taxonomy page to content items.

        Args:
            taxonomy: Taxonomy page to match.
            limit: Maximum number of matches to return.
            min_threshold: Minimum similarity threshold.

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
                0.0,
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
            f"Matched taxonomy {taxonomy.destination_url} to {len(matches)} content items. "
            f"Best match score: {matches[0][1]:.3f}"
            if matches
            else "No matches found"
        )

        return matches

    def find_best_match(
        self,
        content: WordPressContent,
        min_threshold: float | None = None,
        taxonomy_pool: dict[UUID, TaxonomyPage] | None = None,
    ) -> tuple[TaxonomyPage, float] | None:
        """Find best matching taxonomy for a content item.

        Args:
            content: Content item.
            min_threshold: Minimum similarity threshold (uses settings default if None).

        Returns:
            (taxonomy, score) tuple if match found above threshold, None otherwise.
        """
        threshold = (
            min_threshold if min_threshold is not None else self.settings.similarity_threshold
        )

        matches = self.match_taxonomy_to_content(
            content,
            limit=self.settings.semantic_candidate_limit,
            min_threshold=min_threshold,
            taxonomy_pool=taxonomy_pool,
        )

        if matches:
            top = matches[0]
            if top[1] < threshold:
                logger.debug(
                    "Top semantic candidate for %s below threshold %.2f (score=%.3f)",
                    content.url,
                    threshold,
                    top[1],
                )
            return top

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

        taxonomy_pages = self.db.get_all_taxonomy()
        matched_rows = self.db.get_all_matchings()
        accepted_stages = {MatchStage.SEMANTIC_MATCHED, MatchStage.LLM_CATEGORIZED}

        matched_taxonomy_ids = {
            row.taxonomy_id
            for row in matched_rows
            if row.taxonomy_id is not None
            and row.match_stage in accepted_stages
            and (
                row.match_stage != MatchStage.SEMANTIC_MATCHED
                or row.semantic_similarity_score >= min_threshold
            )
        }

        unmatched = [tax for tax in taxonomy_pages if tax.id not in matched_taxonomy_ids]
        logger.info(
            "Found %s taxonomy page(s) without accepted matches (threshold %.2f)",
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
        """Match all content items to taxonomy pages.

        Args:
            taxonomy_pages: Optional list of taxonomy pages (for filtering candidates).
            content_items: Optional list of content items. If None, loads from database.
            min_threshold: Minimum similarity threshold.
            store_results: Whether to store results in database.

        Returns:
            Dictionary mapping content_id to best matching result (or None).
        """
        if min_threshold is None:
            min_threshold = self.settings.similarity_threshold

        taxonomy_lookup: dict[UUID, TaxonomyPage] | None = None
        target_taxonomy_ids: set[UUID] | None = None
        existing_matches_lookup: dict[UUID, MatchingResult] = {}
        if taxonomy_pages is not None:
            taxonomy_lookup = {tax.id: tax for tax in taxonomy_pages}
            target_taxonomy_ids = set(taxonomy_lookup.keys())
            logger.info(
                "Restricting taxonomy search to %s row(s)",
                len(taxonomy_lookup),
            )
            try:
                existing_matches = self.db.get_all_matchings()
                existing_matches_lookup = {row.content_id: row for row in existing_matches}
            except Exception as exc:  # pragma: no cover - monitoring only
                logger.warning("Failed to load existing matchings for subset filtering: %s", exc)

        # Get all content items if not provided
        content_items = content_items or self.db.get_all_content()
        logger.info(
            "Matching %s content items via stored embeddings",
            len(content_items),
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

        for content in content_items:
            # Ensure content embedding exists
            self._ensure_content_embedding(content)

            # Find best matching taxonomy for this content
            best_match = self.find_best_match(content, min_threshold, taxonomy_lookup)
            candidate_taxonomy: TaxonomyPage | None = None
            candidate_score = 0.0
            if best_match:
                candidate_taxonomy, candidate_score = best_match

            if target_taxonomy_ids is not None:
                existing_record = existing_matches_lookup.get(content.id)
                candidate_id = candidate_taxonomy.id if candidate_taxonomy else None
                if not self._content_relates_to_targets(candidate_id, existing_record, target_taxonomy_ids):
                    logger.debug(
                        "Skipping content %s; no overlap with targeted taxonomy subset",
                        content.url,
                    )
                    continue

            semantic_taxonomy_id = candidate_taxonomy.id if candidate_taxonomy else None

            if candidate_taxonomy and candidate_score >= min_threshold:
                matching_result = MatchingResult(
                    taxonomy_id=candidate_taxonomy.id,
                    content_id=content.id,
                    semantic_taxonomy_id=semantic_taxonomy_id,
                    semantic_similarity_score=candidate_score,
                    match_stage=MatchStage.SEMANTIC_MATCHED,
                    updated_at=datetime.now(timezone.utc),
                )
                logger.info(
                    "Semantic match ✔ %s → %s (score %.3f)",
                    content.url,
                    candidate_taxonomy.destination_url,
                    candidate_score,
                )
            else:
                if candidate_taxonomy:
                    logger.debug(
                        "Content %s best semantic candidate %s below threshold %.2f (score %.3f)",
                        content.url,
                        candidate_taxonomy.destination_url,
                        min_threshold,
                        candidate_score,
                    )
                else:
                    logger.warning("No semantic candidates found for content %s", content.url)

                matching_result = MatchingResult(
                    taxonomy_id=None,
                    content_id=content.id,
                    semantic_taxonomy_id=semantic_taxonomy_id,
                    semantic_similarity_score=candidate_score if candidate_taxonomy else 0.0,
                    match_stage=MatchStage.NEEDS_LLM_REVIEW,
                    failed_at_stage="semantic_matching",
                    updated_at=datetime.now(timezone.utc),
                )

            if store_results:
                pending.append(matching_result)
                if len(pending) >= self.settings.matching_batch_size:
                    flush_pending()

            results[content.id] = matching_result

        flush_pending()

        matched_count = sum(
            1 for result in results.values() if result.match_stage == MatchStage.SEMANTIC_MATCHED
        )
        logger.info(
            "Completed semantic matching: %s/%s content items at ≥ %.2f",
            matched_count,
            len(content_items),
            min_threshold,
        )

        return results

    @staticmethod
    def _content_relates_to_targets(
        candidate_taxonomy_id: UUID | None,
        existing_match: MatchingResult | None,
        target_ids: set[UUID],
    ) -> bool:
        if candidate_taxonomy_id and candidate_taxonomy_id in target_ids:
            return True
        if existing_match:
            if existing_match.taxonomy_id and existing_match.taxonomy_id in target_ids:
                return True
            if (
                existing_match.semantic_taxonomy_id
                and existing_match.semantic_taxonomy_id in target_ids
            ):
                return True
        return False

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
        delegates to ``match_all_taxonomy`` to avoid duplicate logic while still
        honoring caller-provided subsets.
        """

        return self.match_all_taxonomy(
            taxonomy_pages=taxonomy_pages,
            content_items=content_items,
            min_threshold=min_threshold,
            store_results=store_results,
        )

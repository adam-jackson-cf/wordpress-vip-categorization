"""Semantic matching service for taxonomy to content mapping."""

import logging
from uuid import UUID

import numpy as np
import openai
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from src.config import Settings
from src.data.supabase_client import SupabaseClient
from src.models import MatchingResult, MatchStage, TaxonomyPage, WordPressContent

logger = logging.getLogger(__name__)


class MatchingService:
    """Service for semantic matching between taxonomy and content.

    Uses OpenAI embeddings to compute semantic similarity between
    taxonomy pages (with keywords/descriptions) and WordPress content.
    """

    def __init__(self, settings: Settings, db_client: SupabaseClient) -> None:
        """Initialize matching service.

        Args:
            settings: Application settings.
            db_client: Supabase database client.
        """
        self.settings = settings
        self.db = db_client
        self.client = openai.OpenAI(
            api_key=settings.semantic_api_key, base_url=settings.semantic_base_url
        )
        self.embedding_model = settings.semantic_embedding_model
        logger.info(
            f"Initialized matching service with model: {self.embedding_model}, "
            f"base URL: {settings.semantic_base_url}"
        )

    @retry(
        retry=retry_if_exception_type(openai.APIError),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
    )
    def get_embedding(self, text: str) -> list[float]:
        """Get embedding vector for text.

        Args:
            text: Text to embed.

        Returns:
            Embedding vector.
        """
        response = self.client.embeddings.create(
            model=self.embedding_model, input=text, encoding_format="float"
        )
        return response.data[0].embedding

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

    def match_taxonomy_to_content(
        self, taxonomy: TaxonomyPage, content_items: list[WordPressContent]
    ) -> list[tuple[WordPressContent, float]]:
        """Match a taxonomy page to content items.

        Args:
            taxonomy: Taxonomy page to match.
            content_items: List of content items to match against.

        Returns:
            List of (content, similarity_score) tuples, sorted by score descending.
        """
        # Get taxonomy embedding
        taxonomy_text = self.create_taxonomy_text(taxonomy)
        taxonomy_embedding = self.get_embedding(taxonomy_text)

        matches = []
        for content in content_items:
            # Get content embedding
            content_text = self.create_content_text(content)
            content_embedding = self.get_embedding(content_text)

            # Compute similarity
            similarity = self.compute_similarity(taxonomy_embedding, content_embedding)
            matches.append((content, similarity))

        # Sort by similarity descending
        matches.sort(key=lambda x: x[1], reverse=True)

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
        content_items: list[WordPressContent],
        min_threshold: float | None = None,
    ) -> tuple[WordPressContent, float] | None:
        """Find best matching content for a taxonomy page.

        Args:
            taxonomy: Taxonomy page.
            content_items: Content items to match against.
            min_threshold: Minimum similarity threshold (uses settings default if None).

        Returns:
            (content, score) tuple if match found above threshold, None otherwise.
        """
        if min_threshold is None:
            min_threshold = self.settings.similarity_threshold

        matches = self.match_taxonomy_to_content(taxonomy, content_items)

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

        # Get all taxonomy pages
        taxonomy_pages = self.db.get_all_taxonomy()

        # Get all matching results
        all_matches = self.db.get_all_matchings()

        # Build map of taxonomy_id to matching result
        matches_map = {match.taxonomy_id: match for match in all_matches}

        # Filter for unmatched or below threshold
        unmatched = []
        for taxonomy in taxonomy_pages:
            match = matches_map.get(taxonomy.id)
            if not match or not match.content_id or match.similarity_score < min_threshold:
                unmatched.append(taxonomy)

        logger.info(
            f"Found {len(unmatched)}/{len(taxonomy_pages)} taxonomy pages "
            f"below threshold {min_threshold}"
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

        # Get all taxonomy pages and content if not provided
        if taxonomy_pages is None:
            taxonomy_pages = self.db.get_all_taxonomy()
        if content_items is None:
            content_items = self.db.get_all_content()

        logger.info(
            f"Matching {len(taxonomy_pages)} taxonomy pages to {len(content_items)} content items"
        )

        results: dict[UUID, MatchingResult] = {}

        for taxonomy in taxonomy_pages:
            best_match = self.find_best_match(taxonomy, content_items, min_threshold)

            if best_match:
                content, score = best_match
                matching_result = MatchingResult(
                    taxonomy_id=taxonomy.id,
                    content_id=content.id,
                    similarity_score=score,
                    match_stage=MatchStage.SEMANTIC_MATCHED,
                )

                if store_results:
                    self.db.upsert_matching(matching_result)

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
                )

                if store_results:
                    self.db.upsert_matching(matching_result)

                results[taxonomy.id] = matching_result
                logger.warning(
                    f"No match found for taxonomy {taxonomy.url} above threshold {min_threshold}"
                )

        matched_count = len(
            [result for result in results.values() if result.content_id is not None]
        )
        logger.info(
            f"Completed matching: {matched_count} "
            f"out of {len(taxonomy_pages)} taxonomy pages matched"
        )

        return results

    def batch_get_embeddings(self, texts: list[str]) -> list[list[float]]:
        """Get embeddings for multiple texts in batch.

        Args:
            texts: List of texts to embed.

        Returns:
            List of embedding vectors.
        """
        # OpenAI allows up to 2048 inputs per request for embeddings
        batch_size = 2048
        all_embeddings = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            response = self.client.embeddings.create(
                model=self.embedding_model, input=batch, encoding_format="float"
            )
            embeddings = [data.embedding for data in response.data]
            all_embeddings.extend(embeddings)

        return all_embeddings

    def match_all_taxonomy_batch(
        self,
        taxonomy_pages: list[TaxonomyPage] | None = None,
        content_items: list[WordPressContent] | None = None,
        min_threshold: float | None = None,
        store_results: bool = True,
    ) -> dict[UUID, MatchingResult]:
        """Match all taxonomy pages to content using batch embeddings.

        More efficient version that batches embedding requests.

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

        # Get all taxonomy pages and content if not provided
        if taxonomy_pages is None:
            taxonomy_pages = self.db.get_all_taxonomy()
        if content_items is None:
            content_items = self.db.get_all_content()

        logger.info(
            f"Batch matching {len(taxonomy_pages)} taxonomy pages "
            f"to {len(content_items)} content items"
        )

        # Get all embeddings in batch
        taxonomy_texts = [self.create_taxonomy_text(t) for t in taxonomy_pages]
        content_texts = [self.create_content_text(c) for c in content_items]

        logger.info("Computing taxonomy embeddings...")
        taxonomy_embeddings = self.batch_get_embeddings(taxonomy_texts)

        logger.info("Computing content embeddings...")
        content_embeddings = self.batch_get_embeddings(content_texts)

        # Match each taxonomy to all content
        results: dict[UUID, MatchingResult] = {}

        for i, taxonomy in enumerate(taxonomy_pages):
            taxonomy_emb = taxonomy_embeddings[i]
            best_score = 0.0
            best_content = None

            for j, content in enumerate(content_items):
                content_emb = content_embeddings[j]
                similarity = self.compute_similarity(taxonomy_emb, content_emb)

                if similarity > best_score:
                    best_score = similarity
                    best_content = content

            if best_score >= min_threshold and best_content:
                matching_result = MatchingResult(
                    taxonomy_id=taxonomy.id,
                    content_id=best_content.id,
                    similarity_score=best_score,
                    match_stage=MatchStage.SEMANTIC_MATCHED,
                )

                if store_results:
                    self.db.upsert_matching(matching_result)

                results[taxonomy.id] = matching_result
                logger.info(
                    f"Matched taxonomy {taxonomy.url} to {best_content.url} "
                    f"(score: {best_score:.3f})"
                )
            else:
                matching_result = MatchingResult(
                    taxonomy_id=taxonomy.id,
                    content_id=None,
                    similarity_score=best_score,
                    match_stage=MatchStage.NEEDS_HUMAN_REVIEW,
                    failed_at_stage="semantic_matching",
                )

                if store_results:
                    self.db.upsert_matching(matching_result)

                results[taxonomy.id] = matching_result
                logger.warning(
                    f"No match found for taxonomy {taxonomy.url} "
                    f"(best score: {best_score:.3f}, threshold: {min_threshold})"
                )

        matched_count = len(
            [result for result in results.values() if result.content_id is not None]
        )
        logger.info(
            f"Completed batch matching: {matched_count}/{len(taxonomy_pages)} "
            f"taxonomy pages matched"
        )

        return results

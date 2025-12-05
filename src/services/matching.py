"""Semantic matching service for taxonomy to content mapping."""

import logging
import re
import unicodedata
from datetime import datetime, timezone
from urllib.parse import urlparse
from uuid import UUID

import numpy as np

from src.config import Settings
from src.data.supabase_client import SupabaseClient
from src.models import MatchingResult, MatchStage, TaxonomyPage, WordPressContent
from src.services.content_type_detector import (
    CONTENT_TYPE_RULES,
    detect_content_type,
)
from src.services.embeddings import EmbeddingService
from src.services.language import detect_language_code

logger = logging.getLogger(__name__)

SINGLE_AUDIENCE_BONUS = 0.04
SINGLE_SPECIES_BONUS = 0.04
REGULATORY_COMBO_BONUS = 0.05
REGULATORY_AUDIENCES = {"veterinarians", "pet owners"}


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

    @staticmethod
    def _normalize_path_value(url_value: str | None) -> str | None:
        if not url_value:
            return None
        value = url_value.strip()
        if not value:
            return None
        parsed = urlparse(value)
        path = parsed.path
        if parsed.scheme or parsed.netloc:
            path = parsed.path
        elif value.startswith("/"):
            path = value
        else:
            path = f"/{value}"
        segments = [segment for segment in path.split("/") if segment]
        normalized = "/" + "/".join(segments) if segments else "/"
        return normalized.lower()

    def _build_reference_lookup(
        self, taxonomy_pages: list[TaxonomyPage] | None
    ) -> dict[str, TaxonomyPage]:
        lookup: dict[str, TaxonomyPage] = {}
        if not taxonomy_pages:
            return lookup
        for taxonomy in taxonomy_pages:
            normalized = self._normalize_path_value(taxonomy.reference_source)
            if not normalized:
                continue
            lookup.setdefault(normalized, taxonomy)
        return lookup

    def _should_run_url_checker(self, content: WordPressContent) -> bool:
        if not self.settings.enable_url_stage_zero:
            return False
        # TODO: Reinstate category gating if Stage-0 should be restricted to specific items again.
        return True

    def _match_reference_source(
        self,
        content: WordPressContent,
        reference_lookup: dict[str, TaxonomyPage],
    ) -> TaxonomyPage | None:
        if not reference_lookup:
            return None
        normalized = self._normalize_path_value(str(content.url))
        if not normalized:
            return None
        return reference_lookup.get(normalized)

    def _path_tokens(self, url_value: str) -> set[str]:
        normalized = self._tokenize_url_path(url_value).lower()
        return {token for token in re.split(r"[^a-z0-9]+", normalized) if token}

    @staticmethod
    def _normalize_token(value: str | None) -> str:
        if not value:
            return ""
        decomposed = unicodedata.normalize("NFKD", value)
        ascii_only = decomposed.encode("ascii", "ignore").decode("ascii")
        return ascii_only.strip().lower()

    def _get_content_audiences(self, content: WordPressContent) -> set[str]:
        values = content.detected_audiences or content.metadata.get("detected_audiences") or []
        return {self._normalize_token(value) for value in values if value}

    def _get_content_species(self, content: WordPressContent) -> set[str]:
        values = content.detected_species or content.metadata.get("detected_species") or []
        return {self._normalize_token(value) for value in values if value}

    def _audience_compatible(self, taxonomy: TaxonomyPage, content: WordPressContent) -> bool:
        primary = self._normalize_token(taxonomy.primary_audiance)
        secondary = self._normalize_token(taxonomy.secondary_audiance)
        detected = self._get_content_audiences(content)
        if not primary and not secondary:
            return True
        if not detected:
            return False
        if primary and not secondary:
            return primary in detected
        valid = {token for token in (primary, secondary) if token}
        return bool(valid & detected)

    def _species_compatible(self, taxonomy: TaxonomyPage, content: WordPressContent) -> bool:
        if not taxonomy.species:
            return True
        required = {
            self._normalize_token(value)
            for value in taxonomy.species
            if value and value.lower() not in {"n/a", "none"}
        }
        if not required:
            return True
        detected = self._get_content_species(content)
        return bool(detected) and required.issubset(detected)

    def _passes_compliance(self, taxonomy: TaxonomyPage, content: WordPressContent) -> bool:
        return self._audience_compatible(taxonomy, content) and self._species_compatible(
            taxonomy, content
        )

    def _priority_boost(
        self, taxonomy: TaxonomyPage, content: WordPressContent | None = None
    ) -> float:
        """Calculate priority boost for taxonomy-content pairs.

        Rewards compliant matches (audience + species alignment) and
        URL token overlap to improve semantic scoring of valid pairs.

        Args:
            taxonomy: Taxonomy page to evaluate.
            content: Optional content for enhanced bonuses.

        Returns:
            Boost value to add to similarity score.
        """
        bonus = 0.0

        # Base bonuses (existing)
        if taxonomy.primary_audiance or taxonomy.secondary_audiance:
            bonus += 0.01
        if taxonomy.species:
            bonus += 0.02

        # Enhanced bonuses when content is provided
        if content:
            # Combined compliance bonus: reward when BOTH audience AND species align
            if self._audience_compatible(taxonomy, content) and self._species_compatible(
                taxonomy, content
            ):
                bonus += 0.05
                logger.debug(
                    "Compliance bonus +0.05 applied for %s → %s",
                    content.url,
                    taxonomy.destination_url,
                )

            # URL token overlap bonus
            tax_tokens = self._path_tokens(str(taxonomy.destination_url))
            content_tokens = self._path_tokens(str(content.url))

            if tax_tokens and content_tokens:
                overlap_ratio = len(tax_tokens & content_tokens) / len(tax_tokens)
                if overlap_ratio > 0.3:  # Significant overlap threshold
                    bonus += 0.03
                    logger.debug(
                        "URL overlap bonus +0.03 applied for %s → %s (overlap=%.2f)",
                        content.url,
                        taxonomy.destination_url,
                        overlap_ratio,
                    )

            content_audiences = self._get_content_audiences(content)
            content_species = self._get_content_species(content)
            single_audience_bonus_applied = False
            single_species_bonus_applied = False

            required_audiences = {
                token
                for token in (
                    self._normalize_token(taxonomy.primary_audiance),
                    self._normalize_token(taxonomy.secondary_audiance),
                )
                if token
            }
            if (
                required_audiences
                and len(content_audiences) == 1
                and (audience_value := next(iter(content_audiences))) in required_audiences
                and audience_value in REGULATORY_AUDIENCES
            ):
                bonus += SINGLE_AUDIENCE_BONUS
                single_audience_bonus_applied = True
                logger.debug(
                    "Single-audience bonus +%.3f applied for %s → %s (%s)",
                    SINGLE_AUDIENCE_BONUS,
                    content.url,
                    taxonomy.destination_url,
                    audience_value,
                )

            required_species = {
                self._normalize_token(value)
                for value in (taxonomy.species or [])
                if value and value.lower() not in {"n/a", "none"}
            }
            if (
                required_species
                and len(content_species) == 1
                and (species_value := next(iter(content_species))) in required_species
            ):
                bonus += SINGLE_SPECIES_BONUS
                single_species_bonus_applied = True
                logger.debug(
                    "Single-species bonus +%.3f applied for %s → %s (%s)",
                    SINGLE_SPECIES_BONUS,
                    content.url,
                    taxonomy.destination_url,
                    species_value,
                )

            if single_audience_bonus_applied and single_species_bonus_applied:
                bonus += REGULATORY_COMBO_BONUS
                logger.debug(
                    "Regulatory combo bonus +%.3f applied for %s → %s",
                    REGULATORY_COMBO_BONUS,
                    content.url,
                    taxonomy.destination_url,
                )

            content_type_hint = self._infer_content_type_hint(content)
            if (
                self.settings.enable_content_type_hinting
                and content_type_hint
                and taxonomy.content_type
            ):
                normalized_taxonomy_type = self._normalize_token(taxonomy.content_type)
                normalized_hint = self._normalize_token(content_type_hint)
                if normalized_taxonomy_type and normalized_hint == normalized_taxonomy_type:
                    rule = CONTENT_TYPE_RULES.get(content_type_hint)
                    hint_bonus = rule.bonus if rule else 0.02
                    bonus += hint_bonus
                    logger.debug(
                        "Content-type hint bonus +%.3f applied for %s → %s (%s)",
                        hint_bonus,
                        content.url,
                        taxonomy.destination_url,
                        content_type_hint,
                    )

        return bonus

    def _filter_taxonomy_candidates(
        self,
        content: WordPressContent,
        matches: list[tuple[TaxonomyPage, float]],
    ) -> list[tuple[TaxonomyPage, float]]:
        filtered: list[tuple[TaxonomyPage, float]] = []
        for taxonomy, score in matches:
            if not self._passes_compliance(taxonomy, content):
                continue

            boosted_score = min(1.0, score + self._priority_boost(taxonomy, content))

            # Debug log for compliant but low-scoring pairs
            if boosted_score < self.settings.similarity_threshold:
                logger.debug(
                    "Compliant candidate fell below threshold: content=%s, taxonomy=%s, "
                    "score=%.3f, boosted=%.3f, threshold=%.2f, "
                    "detected_aud=%s, detected_spc=%s, required_aud=%s/%s, required_spc=%s",
                    content.url,
                    taxonomy.destination_url,
                    score,
                    boosted_score,
                    self.settings.similarity_threshold,
                    self._get_content_audiences(content),
                    self._get_content_species(content),
                    taxonomy.primary_audiance,
                    taxonomy.secondary_audiance,
                    taxonomy.species,
                )

            filtered.append((taxonomy, boosted_score))
        filtered.sort(key=lambda item: item[1], reverse=True)
        return filtered

    def _filter_content_candidates(
        self,
        taxonomy: TaxonomyPage,
        matches: list[tuple[WordPressContent, float]],
    ) -> list[tuple[WordPressContent, float]]:
        filtered: list[tuple[WordPressContent, float]] = []
        for content, score in matches:
            if not self._passes_compliance(taxonomy, content):
                continue
            # Pass content to enable enhanced bonuses
            filtered.append((content, min(1.0, score + self._priority_boost(taxonomy, content))))
        filtered.sort(key=lambda item: item[1], reverse=True)
        return filtered

    def _describe_audiences(self, taxonomy: TaxonomyPage) -> str:
        audiences: list[str] = []
        if taxonomy.primary_audiance:
            audiences.append(f"Primary: {taxonomy.primary_audiance}")
        if taxonomy.secondary_audiance:
            audiences.append(f"Secondary: {taxonomy.secondary_audiance}")
        return ", ".join(audiences)

    @staticmethod
    def _infer_content_type_hint(content: WordPressContent) -> str | None:
        metadata_hint = (content.metadata or {}).get("content_type_hint")
        if metadata_hint:
            return str(metadata_hint)
        slug_value = (content.metadata or {}).get("slug")
        return detect_content_type(str(content.url), slug_value)

    def get_embedding(self, text: str) -> list[float]:
        """Get embedding vector for text using the shared embedding service."""

        return self.embedding_service.embed(text)

    def create_taxonomy_text(self, taxonomy: TaxonomyPage) -> str:
        """Create text representation of taxonomy page for embedding.

        Priority fields (destination URL, local name, key topics) are emphasized
        through duplication and explicit markers to improve semantic matching.

        Args:
            taxonomy: Taxonomy page.

        Returns:
            Combined text for embedding.
        """
        primary_audience = taxonomy.primary_audiance or "unspecified"
        secondary_audience = taxonomy.secondary_audiance or "none"
        species_sentence = ", ".join(taxonomy.species) if taxonomy.species else "unknown"

        # Emphasize key topics through duplication
        key_topics_list = taxonomy.key_topics or []
        topics_emphasized = " | ".join(key_topics_list) if key_topics_list else ""

        # Priority-first ordering with duplication for weight
        parts = [
            # Priority Field 1: Destination URL (duplicate for emphasis)
            f"Priority Field - Destination Path: {self._tokenize_url_path(str(taxonomy.destination_url))}",
            f"Destination Path: {self._tokenize_url_path(str(taxonomy.destination_url))}",
            # Priority Field 2: Local Page Name (duplicate for emphasis)
            (
                f"Priority Field - Local Name: {taxonomy.local_page_name}"
                if taxonomy.local_page_name
                else None
            ),
            f"Local Name: {taxonomy.local_page_name}" if taxonomy.local_page_name else None,
            # Priority Field 3: Key Topics (duplicate for weight)
            f"Key Topics (Primary): {topics_emphasized}" if topics_emphasized else None,
            f"Key Topics (Secondary): {', '.join(key_topics_list)}" if key_topics_list else None,
            # Priority Field 4: Audiences (duplicate for alignment with content)
            f"Priority Field - Primary Audience: {primary_audience}",
            f"Primary Audience: {primary_audience}",
            f"Priority Field - Secondary Audience: {secondary_audience}",
            f"Secondary Audience: {secondary_audience}",
            # Priority Field 5: Species (duplicate for alignment with content)
            f"Priority Field - Species: {species_sentence}",
            f"Species: {species_sentence}",
            # Summary last (previously dominant, now deprioritized)
            f"Summary: {taxonomy.semantic_summary}",
        ]

        return "\n".join(part for part in parts if part)

    def create_content_text(self, content: WordPressContent) -> str:
        """Create text representation of content for embedding.

        URL path, title, slug, and detection signals are emphasized through
        duplication to align better with reweighted taxonomy embeddings.

        Args:
            content: WordPress content.

        Returns:
            Combined text for embedding.
        """
        metadata = content.metadata or {}
        slug = metadata.get("slug")
        categories = [str(value) for value in (metadata.get("categories") or [])]
        tags = [str(value) for value in (metadata.get("tags") or [])]

        # Optimize excerpt length
        excerpt = (metadata.get("excerpt") or "").strip()
        primary_excerpt = excerpt or content.content[:400]
        excerpt_line = f"Excerpt: {primary_excerpt[:400]}" if primary_excerpt else None

        # Truncate preview to 4000 chars to retain more catalog copy for detection cues
        preview = content.content[:4000]
        language_code = detect_language_code(preview or content.title)

        detected_audiences = ", ".join(sorted(self._get_content_audiences(content))) or "unknown"
        detected_species = ", ".join(sorted(self._get_content_species(content))) or "unknown"

        # Priority-first ordering with duplication
        parts = [
            # Priority Field 1: Title (duplicate for weight)
            f"Title (Primary): {content.title}",
            f"Title (Secondary): {content.title}",
            # Priority Field 2: URL Path (duplicate with emphasis)
            f"Priority Field - URL Path: {self._tokenize_url_path(str(content.url))}",
            f"URL Path: {self._tokenize_url_path(str(content.url))}",
            # Priority Field 3: Slug (duplicate for weight)
            f"Slug (Primary): {slug}" if slug else None,
            f"Slug (Secondary): {slug}" if slug else None,
            # Priority Field 4: Detection Signals (duplicate for emphasis)
            f"Priority Field - Detected Audiences: {detected_audiences}",
            f"Detected Audiences: {detected_audiences}",
            f"Priority Field - Detected Species: {detected_species}",
            f"Detected Species: {detected_species}",
            # Supporting metadata
            f"Site: {content.site_url}",
            f"Detected Language: {language_code}",
            excerpt_line,
            f"Categories: {', '.join(categories)}" if categories else None,
            f"Tags: {', '.join(tags)}" if tags else None,
            (
                f"Published: {content.published_date.date().isoformat()}"
                if content.published_date
                else None
            ),
            # Content preview last (truncated to 1000 chars)
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
            raw_matches = self.db.match_taxonomy_by_embedding(
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
            raw_matches = self._local_similarity_search_taxonomy(
                content,
                content_embedding,
                limit,
                taxonomy_pool,
            )

        matches = self._filter_taxonomy_candidates(content, raw_matches)

        if not matches:
            matches = self._filter_taxonomy_candidates(
                content,
                self._local_similarity_search_taxonomy(
                    content,
                    content_embedding,
                    limit,
                    taxonomy_pool,
                ),
            )

        if allowed_ids is not None:
            filtered = [(tax, score) for tax, score in matches if tax.id in allowed_ids]
            if filtered:
                matches = filtered
            else:
                matches = self._filter_taxonomy_candidates(
                    content,
                    self._local_similarity_search_taxonomy(
                        content,
                        content_embedding,
                        limit,
                        taxonomy_pool,
                    ),
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
            filtered = [
                taxonomy
                for taxonomy, score in matches
                if score >= self.settings.llm_candidate_min_score
            ]
            if not filtered:
                filtered = [taxonomy for taxonomy, _ in matches][
                    : self.settings.llm_candidate_limit
                ]
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
            raw_matches = self.db.match_content_by_embedding(
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
            raw_matches = self._local_similarity_search(taxonomy, taxonomy_embedding, limit)

        matches = self._filter_content_candidates(taxonomy, raw_matches)

        if not matches:
            matches = self._filter_content_candidates(
                taxonomy, self._local_similarity_search(taxonomy, taxonomy_embedding, limit)
            )

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
        accepted_stages = {
            MatchStage.URL_MATCHING,
            MatchStage.SEMANTIC_MATCHED,
            MatchStage.LLM_CATEGORIZED,
        }

        matched_taxonomy_ids = {
            row.taxonomy_id
            for row in matched_rows
            if row.taxonomy_id is not None
            and row.match_stage in accepted_stages
            and (
                row.match_stage not in {MatchStage.SEMANTIC_MATCHED}
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

        reference_lookup: dict[str, TaxonomyPage] = {}
        if self.settings.enable_url_stage_zero:
            reference_rows = taxonomy_pages
            if reference_rows is None:
                try:
                    reference_rows = self.db.get_all_taxonomy()
                except Exception as exc:  # pragma: no cover - monitoring only
                    logger.warning("Failed to load taxonomy for URL matching stage: %s", exc)
                    reference_rows = None
            reference_lookup = self._build_reference_lookup(reference_rows)

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
            should_run_url_checker = self._should_run_url_checker(content)
            if should_run_url_checker:
                direct_taxonomy = self._match_reference_source(content, reference_lookup)
                if direct_taxonomy:
                    matching_result = MatchingResult(
                        taxonomy_id=direct_taxonomy.id,
                        content_id=content.id,
                        semantic_taxonomy_id=direct_taxonomy.id,
                        semantic_similarity_score=1.0,
                        match_stage=MatchStage.URL_MATCHING,
                        updated_at=datetime.now(timezone.utc),
                    )
                    logger.info(
                        "URL match ✔ %s → %s (reference_source)",
                        content.url,
                        direct_taxonomy.reference_source or direct_taxonomy.destination_url,
                    )
                    if store_results:
                        pending.append(matching_result)
                        if len(pending) >= self.settings.matching_batch_size:
                            flush_pending()
                    results[content.id] = matching_result
                    continue
                logger.debug(
                    "URL checker miss for %s; proceeding with semantic stage",
                    content.url,
                )

            # Ensure content embedding exists for semantic stage
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
                if not self._content_relates_to_targets(
                    candidate_id, existing_record, target_taxonomy_ids
                ):
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
            1
            for result in results.values()
            if result.match_stage in {MatchStage.URL_MATCHING, MatchStage.SEMANTIC_MATCHED}
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

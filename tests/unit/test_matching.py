"""Unit tests for matching service."""

from unittest.mock import Mock
from uuid import uuid4

import pytest

from src.config import Settings
from src.models import MatchingResult, MatchStage, TaxonomyPage, WordPressContent
from src.services.matching import MatchingService


class TestMatchingService:
    """Tests for matching service."""

    def test_init(self, mock_settings: Settings, mock_supabase_client: Mock) -> None:
        """Test service initialization."""
        service = MatchingService(mock_settings, mock_supabase_client)
        assert service.settings == mock_settings
        assert service.db == mock_supabase_client
        assert service.embedding_model == "text-embedding-3-small"

    def test_create_taxonomy_text(
        self,
        mock_settings: Settings,
        mock_supabase_client: Mock,
        sample_taxonomy_page: TaxonomyPage,
    ) -> None:
        """Test taxonomy text creation with priority-first structure."""
        service = MatchingService(mock_settings, mock_supabase_client)

        text = service.create_taxonomy_text(sample_taxonomy_page)

        # Verify priority fields appear first
        assert text.index("Priority Field - Destination Path") < text.index("Summary")
        assert text.index("Priority Field - Local Name") < text.index("Summary")

        # Verify duplication
        assert text.count("Destination Path") == 2
        assert text.count(sample_taxonomy_page.local_page_name) >= 2
        assert text.count("Primary Audience") == 2
        assert text.count("Secondary Audience") == 2
        assert text.count("Species") == 2

        # Verify key topics ahead of summary
        assert text.index("Key Topics (Primary)") < text.index("Summary")

        # Verify priority markers for audiences and species
        assert "Priority Field - Primary Audience" in text
        assert "Priority Field - Secondary Audience" in text
        assert "Priority Field - Species" in text

        # Existing assertions still valid
        assert "Summary: Guías veterinarias" in text
        assert "Veterinarians" in text
        assert "Producers" in text
        assert "Swine" in text

    def test_create_taxonomy_text_marks_missing_secondary(
        self,
        mock_settings: Settings,
        mock_supabase_client: Mock,
        sample_taxonomy_page: TaxonomyPage,
    ) -> None:
        service = MatchingService(mock_settings, mock_supabase_client)
        sample_taxonomy_page.secondary_audiance = None

        text = service.create_taxonomy_text(sample_taxonomy_page)

        assert "Secondary Audience: none" in text

    def test_create_content_text(
        self,
        mock_settings: Settings,
        mock_supabase_client: Mock,
        sample_wordpress_content: WordPressContent,
    ) -> None:
        """Test content text creation with priority-first structure."""
        service = MatchingService(mock_settings, mock_supabase_client)

        text = service.create_content_text(sample_wordpress_content)

        # Verify priority fields appear first
        assert text.index("Title (Primary)") < text.index("Content Preview")
        assert text.index("Priority Field - URL Path") < text.index("Content Preview")

        # Verify duplication
        assert text.count(sample_wordpress_content.title) >= 2
        assert text.count("Detected Audiences") == 2
        assert text.count("Detected Species") == 2

        # Verify detection emphasis
        assert "Priority Field - Detected Audiences" in text
        assert "Priority Field - Detected Species" in text
        assert "Content Preview:" in text
        assert "Excerpt:" in text

    def test_compute_similarity(self, mock_settings: Settings, mock_supabase_client: Mock) -> None:
        """Test similarity computation."""
        service = MatchingService(mock_settings, mock_supabase_client)

        # Identical vectors should have similarity close to 1
        vec1 = [1.0, 0.0, 0.0]
        vec2 = [1.0, 0.0, 0.0]
        similarity = service.compute_similarity(vec1, vec2)
        assert similarity > 0.99

        # Orthogonal vectors should have similarity around 0.5
        vec3 = [1.0, 0.0, 0.0]
        vec4 = [0.0, 1.0, 0.0]
        similarity = service.compute_similarity(vec3, vec4)
        assert 0.4 < similarity < 0.6

    def test_get_embedding(
        self,
        mock_settings: Settings,
        mock_supabase_client: Mock,
        mock_openai_client: Mock,
    ) -> None:
        """Test embedding retrieval."""

        service = MatchingService(mock_settings, mock_supabase_client)

        embedding = service.get_embedding("test text")

        assert isinstance(embedding, list)
        assert len(embedding) == 1536  # Default embedding size
        service.embedding_service.embed.assert_called_once_with("test text")

    def test_match_taxonomy_to_content(
        self,
        mock_settings: Settings,
        mock_supabase_client: Mock,
        mock_openai_client: Mock,
        sample_taxonomy_page: TaxonomyPage,
        sample_wordpress_content: WordPressContent,
    ) -> None:
        """Test matching content to taxonomy."""
        service = MatchingService(mock_settings, mock_supabase_client)
        mock_supabase_client.match_taxonomy_by_embedding.return_value = [
            (sample_taxonomy_page, 0.9)
        ]

        matches = service.match_taxonomy_to_content(sample_wordpress_content)

        assert len(matches) == 1
        taxonomy, score = matches[0]
        assert taxonomy.id == sample_taxonomy_page.id
        assert 0.0 <= score <= 1.0

    def test_match_taxonomy_to_content_filters_subset(
        self,
        mock_settings: Settings,
        mock_supabase_client: Mock,
        mock_openai_client: Mock,
        sample_taxonomy_page: TaxonomyPage,
        sample_wordpress_content: WordPressContent,
    ) -> None:
        """Ensure only allowed taxonomy rows are returned when subset provided."""

        second_taxonomy = TaxonomyPage(
            id=uuid4(),
            uid="TAX-ALT",
            destination_url="https://taxonomy.com/alt",
            english_page_name="Alt",
            local_page_name="Alt",
            content_type="Alt",
            primary_audiance="All",
            secondary_audiance="All",
            semantic_summary="Alt",
            key_topics=[],
        )

        service = MatchingService(mock_settings, mock_supabase_client)
        mock_supabase_client.match_taxonomy_by_embedding.return_value = [
            (second_taxonomy, 0.95),
            (sample_taxonomy_page, 0.9),
        ]

        matches = service.match_taxonomy_to_content(
            sample_wordpress_content,
            taxonomy_pool={sample_taxonomy_page.id: sample_taxonomy_page},
        )

        assert len(matches) == 1
        assert matches[0][0].id == sample_taxonomy_page.id

    def test_match_taxonomy_to_content_falls_back_when_subset_filters_all(
        self,
        mock_settings: Settings,
        mock_supabase_client: Mock,
        mock_openai_client: Mock,
        sample_taxonomy_page: TaxonomyPage,
        sample_wordpress_content: WordPressContent,
    ) -> None:
        """When Supabase returns no allowed rows, local search should run."""

        disallowed = TaxonomyPage(
            id=uuid4(),
            uid="TAX-DENY",
            destination_url="https://taxonomy.com/disallowed",
            english_page_name="Nope",
            local_page_name="Nope",
            content_type="Nope",
            primary_audiance="All",
            secondary_audiance="All",
            semantic_summary="Nope",
            key_topics=[],
        )

        service = MatchingService(mock_settings, mock_supabase_client)
        mock_supabase_client.match_taxonomy_by_embedding.return_value = [(disallowed, 0.99)]
        service._local_similarity_search_taxonomy = Mock(  # type: ignore[attr-defined]
            return_value=[(sample_taxonomy_page, 0.8)]
        )

        matches = service.match_taxonomy_to_content(
            sample_wordpress_content,
            taxonomy_pool={sample_taxonomy_page.id: sample_taxonomy_page},
        )

        service._local_similarity_search_taxonomy.assert_called_once()
        assert matches[0][0].id == sample_taxonomy_page.id

    def test_match_taxonomy_to_content_requires_audience_alignment(
        self,
        mock_settings: Settings,
        mock_supabase_client: Mock,
        sample_taxonomy_page: TaxonomyPage,
        sample_wordpress_content: WordPressContent,
    ) -> None:
        sample_taxonomy_page.secondary_audiance = None
        sample_wordpress_content.detected_audiences = ["producers"]
        service = MatchingService(mock_settings, mock_supabase_client)
        mock_supabase_client.match_taxonomy_by_embedding.return_value = [
            (sample_taxonomy_page, 0.95)
        ]
        service._local_similarity_search_taxonomy = Mock(return_value=[])  # type: ignore[attr-defined]

        matches = service.match_taxonomy_to_content(sample_wordpress_content)

        assert matches == []

    def test_match_taxonomy_to_content_requires_species_alignment(
        self,
        mock_settings: Settings,
        mock_supabase_client: Mock,
        sample_taxonomy_page: TaxonomyPage,
        sample_wordpress_content: WordPressContent,
    ) -> None:
        sample_taxonomy_page.species = ["bovine"]
        sample_wordpress_content.detected_species = ["swine"]
        service = MatchingService(mock_settings, mock_supabase_client)
        mock_supabase_client.match_taxonomy_by_embedding.return_value = [
            (sample_taxonomy_page, 0.9)
        ]
        service._local_similarity_search_taxonomy = Mock(return_value=[])  # type: ignore[attr-defined]

        matches = service.match_taxonomy_to_content(sample_wordpress_content)

        assert matches == []

    def test_match_all_taxonomy_subset_skips_unrelated_content(
        self,
        mock_settings: Settings,
        mock_supabase_client: Mock,
        mock_openai_client: Mock,
        sample_taxonomy_page: TaxonomyPage,
        sample_wordpress_content: WordPressContent,
    ) -> None:
        """Scoped runs should not demote content tied to other taxonomy rows."""

        other_taxonomy = TaxonomyPage(
            id=uuid4(),
            uid="TAX-OTHER",
            destination_url="https://taxonomy.com/other",
            english_page_name="Other",
            local_page_name="Otro",
            content_type="Other",
            primary_audiance="All",
            secondary_audiance="All",
            semantic_summary="Other",
            key_topics=["misc"],
        )
        other_content = WordPressContent(
            id=uuid4(),
            url="https://example.com/other",
            title="Other Content",
            content="Body",
            site_url="https://example.com",
        )

        captured: list[list[MatchingResult]] = []

        def _capture(data, chunk_size=None):  # type: ignore[no-untyped-def]
            captured.append(list(data))
            return []

        mock_supabase_client.bulk_upsert_matchings.side_effect = _capture

        service = MatchingService(mock_settings, mock_supabase_client)
        service.find_best_match = Mock(  # type: ignore[attr-defined]
            side_effect=[(sample_taxonomy_page, 0.92), None]
        )

        mock_supabase_client.get_all_matchings.return_value = [
            MatchingResult(
                taxonomy_id=sample_taxonomy_page.id,
                content_id=sample_wordpress_content.id,
                semantic_taxonomy_id=sample_taxonomy_page.id,
                semantic_similarity_score=0.9,
                match_stage=MatchStage.SEMANTIC_MATCHED,
            ),
            MatchingResult(
                taxonomy_id=other_taxonomy.id,
                content_id=other_content.id,
                semantic_taxonomy_id=other_taxonomy.id,
                semantic_similarity_score=0.88,
                match_stage=MatchStage.SEMANTIC_MATCHED,
            ),
        ]

        results = service.match_all_taxonomy(
            taxonomy_pages=[sample_taxonomy_page],
            content_items=[sample_wordpress_content, other_content],
        )

        assert sample_wordpress_content.id in results
        assert other_content.id not in results
        mock_supabase_client.bulk_upsert_matchings.assert_called_once()
        assert captured and captured[0][0].content_id == sample_wordpress_content.id

    def test_find_best_match_above_threshold(
        self,
        mock_settings: Settings,
        mock_supabase_client: Mock,
        mock_openai_client: Mock,
        sample_taxonomy_page: TaxonomyPage,
        sample_wordpress_content: WordPressContent,
    ) -> None:
        """Test finding best match above threshold."""
        service = MatchingService(mock_settings, mock_supabase_client)
        mock_supabase_client.match_taxonomy_by_embedding.return_value = [
            (sample_taxonomy_page, 0.8)
        ]

        match = service.find_best_match(sample_wordpress_content, min_threshold=0.0)

        assert match is not None
        taxonomy, score = match
        assert taxonomy.id == sample_taxonomy_page.id

    def test_find_best_match_below_threshold(
        self,
        mock_settings: Settings,
        mock_supabase_client: Mock,
        mock_openai_client: Mock,
        sample_taxonomy_page: TaxonomyPage,
        sample_wordpress_content: WordPressContent,
    ) -> None:
        """Ensure best candidate is returned with enhanced bonuses."""
        service = MatchingService(mock_settings, mock_supabase_client)
        mock_supabase_client.match_taxonomy_by_embedding.return_value = [
            (sample_taxonomy_page, 0.5)
        ]

        match = service.find_best_match(sample_wordpress_content, min_threshold=0.99)

        assert match is not None
        taxonomy, score = match
        assert taxonomy.id == sample_taxonomy_page.id
        # Score: 0.5 + 0.01 (audience) + 0.02 (species) + 0.05 (compliance) = 0.58
        assert score >= 0.58

    def test_get_unmatched_taxonomy_filters_by_threshold(
        self,
        mock_settings: Settings,
        mock_supabase_client: Mock,
        sample_taxonomy_page: TaxonomyPage,
        sample_wordpress_content: WordPressContent,
    ) -> None:
        """Ensure taxonomy rows with high scores are excluded from unmatched list."""

        second_taxonomy = TaxonomyPage(
            id=uuid4(),
            uid="TAX-SECOND",
            destination_url="https://taxonomy.com/second",
            english_page_name="News Page",
            local_page_name="Pagina Noticias",
            content_type="News",
            primary_audiance="All",
            secondary_audiance="Media",
            semantic_summary="News summary",
            key_topics=[],
        )

        service = MatchingService(mock_settings, mock_supabase_client)

        matched_row = MatchingResult(
            taxonomy_id=sample_taxonomy_page.id,
            semantic_taxonomy_id=sample_taxonomy_page.id,
            content_id=sample_wordpress_content.id,
            semantic_similarity_score=0.92,
            match_stage=MatchStage.SEMANTIC_MATCHED,
        )
        mock_supabase_client.get_all_taxonomy.return_value = [
            sample_taxonomy_page,
            second_taxonomy,
        ]
        mock_supabase_client.get_all_matchings.return_value = [matched_row]
        unmatched = service.get_unmatched_taxonomy(min_threshold=0.85)

        assert unmatched == [second_taxonomy]

    def test_match_all_taxonomy_records_matches(
        self,
        mock_settings: Settings,
        mock_supabase_client: Mock,
        sample_taxonomy_page: TaxonomyPage,
        sample_wordpress_content: WordPressContent,
    ) -> None:
        """match_all_taxonomy should persist results using find_best_match."""

        service = MatchingService(mock_settings, mock_supabase_client)
        service.find_best_match = Mock(return_value=(sample_taxonomy_page, 0.92))

        results = service.match_all_taxonomy(
            taxonomy_pages=[sample_taxonomy_page],
            content_items=[sample_wordpress_content],
            min_threshold=0.8,
        )

        assert sample_wordpress_content.id in results
        match_result = results[sample_wordpress_content.id]
        assert match_result.taxonomy_id == sample_taxonomy_page.id
        assert match_result.content_id == sample_wordpress_content.id
        assert match_result.semantic_taxonomy_id == sample_taxonomy_page.id
        assert match_result.semantic_similarity_score == 0.92
        mock_supabase_client.bulk_upsert_matchings.assert_called()
        service.find_best_match.assert_called_once_with(
            sample_wordpress_content,
            0.8,
            {sample_taxonomy_page.id: sample_taxonomy_page},
        )

    def test_match_all_taxonomy_records_needs_review(
        self,
        mock_settings: Settings,
        mock_supabase_client: Mock,
        sample_taxonomy_page: TaxonomyPage,
        sample_wordpress_content: WordPressContent,
    ) -> None:
        """When no best match exists, a needs-review result should be stored."""

        service = MatchingService(mock_settings, mock_supabase_client)
        service.find_best_match = Mock(return_value=(sample_taxonomy_page, 0.5))

        results = service.match_all_taxonomy(
            taxonomy_pages=[sample_taxonomy_page],
            content_items=[sample_wordpress_content],
            min_threshold=0.95,
        )

        match_result = results[sample_wordpress_content.id]
        assert match_result.match_stage == MatchStage.NEEDS_LLM_REVIEW
        assert match_result.taxonomy_id is None
        assert match_result.content_id == sample_wordpress_content.id
        assert match_result.semantic_taxonomy_id == sample_taxonomy_page.id
        assert match_result.semantic_similarity_score == 0.5
        mock_supabase_client.bulk_upsert_matchings.assert_called_once()
        service.find_best_match.assert_called_once_with(
            sample_wordpress_content,
            0.95,
            {sample_taxonomy_page.id: sample_taxonomy_page},
        )

    def test_match_all_taxonomy_batch_uses_precomputed_embeddings(
        self,
        mock_settings: Settings,
        mock_supabase_client: Mock,
        sample_taxonomy_page: TaxonomyPage,
        sample_wordpress_content: WordPressContent,
    ) -> None:
        """Batch helper should delegate to match_all_taxonomy for vector-backed flow."""

        service = MatchingService(mock_settings, mock_supabase_client)
        delegated = {
            sample_wordpress_content.id: MatchingResult(
                taxonomy_id=sample_taxonomy_page.id,
                content_id=sample_wordpress_content.id,
                semantic_taxonomy_id=sample_taxonomy_page.id,
                semantic_similarity_score=0.9,
                match_stage=MatchStage.SEMANTIC_MATCHED,
            )
        }
        service.match_all_taxonomy = Mock(return_value=delegated)

        results = service.match_all_taxonomy_batch(
            taxonomy_pages=[sample_taxonomy_page],
            content_items=[sample_wordpress_content],
            min_threshold=0.5,
        )

        service.match_all_taxonomy.assert_called_once_with(
            taxonomy_pages=[sample_taxonomy_page],
            content_items=[sample_wordpress_content],
            min_threshold=0.5,
            store_results=True,
        )
        assert results == delegated

    def test_match_all_taxonomy_batch_does_not_reload_content(
        self,
        mock_settings: Settings,
        mock_supabase_client: Mock,
        sample_taxonomy_page: TaxonomyPage,
        sample_wordpress_content: WordPressContent,
    ) -> None:
        """Providing a content subset should skip fetching all content rows."""

        mock_supabase_client.get_all_content.reset_mock()
        service = MatchingService(mock_settings, mock_supabase_client)
        service.find_best_match = Mock(return_value=None)

        service.match_all_taxonomy_batch(
            taxonomy_pages=[sample_taxonomy_page],
            content_items=[sample_wordpress_content],
            min_threshold=0.9,
        )

        mock_supabase_client.get_all_content.assert_not_called()

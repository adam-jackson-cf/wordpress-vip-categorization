"""Unit tests for matching service."""

from unittest.mock import Mock
from uuid import uuid4

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
        """Test taxonomy text creation."""
        service = MatchingService(mock_settings, mock_supabase_client)

        text = service.create_taxonomy_text(sample_taxonomy_page)

        assert "Category: Technology" in text
        assert "Description: Technology-related content" in text
        assert "Keywords: technology, innovation, digital" in text

    def test_create_content_text(
        self,
        mock_settings: Settings,
        mock_supabase_client: Mock,
        sample_wordpress_content: WordPressContent,
    ) -> None:
        """Test content text creation."""
        service = MatchingService(mock_settings, mock_supabase_client)

        text = service.create_content_text(sample_wordpress_content)

        assert "Title:" in text
        assert sample_wordpress_content.title in text
        assert "Content:" in text

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
        """Test matching taxonomy to content."""
        service = MatchingService(mock_settings, mock_supabase_client)
        mock_supabase_client.match_content_by_embedding.return_value = [
            (sample_wordpress_content, 0.9)
        ]

        matches = service.match_taxonomy_to_content(sample_taxonomy_page)

        assert len(matches) == 1
        content, score = matches[0]
        assert content.id == sample_wordpress_content.id
        assert 0.0 <= score <= 1.0

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
        mock_supabase_client.match_content_by_embedding.return_value = [
            (sample_wordpress_content, 0.8)
        ]

        match = service.find_best_match(sample_taxonomy_page, min_threshold=0.0)

        assert match is not None
        content, score = match
        assert content.id == sample_wordpress_content.id

    def test_find_best_match_below_threshold(
        self,
        mock_settings: Settings,
        mock_supabase_client: Mock,
        mock_openai_client: Mock,
        sample_taxonomy_page: TaxonomyPage,
        sample_wordpress_content: WordPressContent,
    ) -> None:
        """Test no match when below threshold."""
        service = MatchingService(mock_settings, mock_supabase_client)
        mock_supabase_client.match_content_by_embedding.return_value = [
            (sample_wordpress_content, 0.5)
        ]

        match = service.find_best_match(sample_taxonomy_page, min_threshold=0.99)

        # Should return None if no match above threshold
        # (depends on mock embedding similarity)
        assert match is None or match[1] >= 0.99

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
            url="https://taxonomy.com/second",
            category="News",
            description="News",
            keywords=[],
        )

        mock_supabase_client.get_unmatched_taxonomy.return_value = [second_taxonomy]

        service = MatchingService(mock_settings, mock_supabase_client)
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
        service.find_best_match = Mock(return_value=(sample_wordpress_content, 0.92))

        results = service.match_all_taxonomy(
            taxonomy_pages=[sample_taxonomy_page],
            content_items=[sample_wordpress_content],
            min_threshold=0.8,
        )

        assert sample_taxonomy_page.id in results
        mock_supabase_client.bulk_upsert_matchings.assert_called()

    def test_match_all_taxonomy_records_needs_review(
        self,
        mock_settings: Settings,
        mock_supabase_client: Mock,
        sample_taxonomy_page: TaxonomyPage,
        sample_wordpress_content: WordPressContent,
    ) -> None:
        """When no best match exists, a needs-review result should be stored."""

        service = MatchingService(mock_settings, mock_supabase_client)
        service.find_best_match = Mock(return_value=None)

        results = service.match_all_taxonomy(
            taxonomy_pages=[sample_taxonomy_page],
            content_items=[sample_wordpress_content],
            min_threshold=0.95,
        )

        match_result = results[sample_taxonomy_page.id]
        assert match_result.match_stage == MatchStage.NEEDS_HUMAN_REVIEW
        mock_supabase_client.bulk_upsert_matchings.assert_called()

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
            sample_taxonomy_page.id: MatchingResult(
                taxonomy_id=sample_taxonomy_page.id,
                content_id=sample_wordpress_content.id,
                similarity_score=0.9,
                match_stage=MatchStage.SEMANTIC_MATCHED,
            )
        }
        service.match_all_taxonomy = Mock(return_value=delegated)

        results = service.match_all_taxonomy_batch(
            taxonomy_pages=[sample_taxonomy_page],
            content_items=[sample_wordpress_content],
            min_threshold=0.5,
        )

        service.match_all_taxonomy.assert_called_once()
        assert results == delegated

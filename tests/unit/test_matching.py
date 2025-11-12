"""Unit tests for matching service."""

from unittest.mock import Mock, patch

from src.config import Settings
from src.models import TaxonomyPage, WordPressContent
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

    @patch("src.services.matching.openai.OpenAI")
    def test_get_embedding(
        self,
        mock_openai_class: Mock,
        mock_settings: Settings,
        mock_supabase_client: Mock,
        mock_openai_client: Mock,
    ) -> None:
        """Test embedding retrieval."""
        mock_openai_class.return_value = mock_openai_client

        service = MatchingService(mock_settings, mock_supabase_client)
        service.client = mock_openai_client

        embedding = service.get_embedding("test text")

        assert isinstance(embedding, list)
        assert len(embedding) == 1536  # Default embedding size
        mock_openai_client.embeddings.create.assert_called_once()

    @patch("src.services.matching.openai.OpenAI")
    def test_match_taxonomy_to_content(
        self,
        mock_openai_class: Mock,
        mock_settings: Settings,
        mock_supabase_client: Mock,
        mock_openai_client: Mock,
        sample_taxonomy_page: TaxonomyPage,
        sample_wordpress_content: WordPressContent,
    ) -> None:
        """Test matching taxonomy to content."""
        mock_openai_class.return_value = mock_openai_client

        service = MatchingService(mock_settings, mock_supabase_client)
        service.client = mock_openai_client

        matches = service.match_taxonomy_to_content(
            sample_taxonomy_page, [sample_wordpress_content]
        )

        assert len(matches) == 1
        content, score = matches[0]
        assert content.id == sample_wordpress_content.id
        assert 0.0 <= score <= 1.0

    @patch("src.services.matching.openai.OpenAI")
    def test_find_best_match_above_threshold(
        self,
        mock_openai_class: Mock,
        mock_settings: Settings,
        mock_supabase_client: Mock,
        mock_openai_client: Mock,
        sample_taxonomy_page: TaxonomyPage,
        sample_wordpress_content: WordPressContent,
    ) -> None:
        """Test finding best match above threshold."""
        mock_openai_class.return_value = mock_openai_client

        service = MatchingService(mock_settings, mock_supabase_client)
        service.client = mock_openai_client

        # Use very low threshold
        match = service.find_best_match(
            sample_taxonomy_page, [sample_wordpress_content], min_threshold=0.0
        )

        assert match is not None
        content, score = match
        assert content.id == sample_wordpress_content.id

    @patch("src.services.matching.openai.OpenAI")
    def test_find_best_match_below_threshold(
        self,
        mock_openai_class: Mock,
        mock_settings: Settings,
        mock_supabase_client: Mock,
        mock_openai_client: Mock,
        sample_taxonomy_page: TaxonomyPage,
        sample_wordpress_content: WordPressContent,
    ) -> None:
        """Test no match when below threshold."""
        mock_openai_class.return_value = mock_openai_client

        service = MatchingService(mock_settings, mock_supabase_client)
        service.client = mock_openai_client

        # Use very high threshold
        match = service.find_best_match(
            sample_taxonomy_page, [sample_wordpress_content], min_threshold=0.99
        )

        # Should return None if no match above threshold
        # (depends on mock embedding similarity)
        assert match is None or match[1] >= 0.99

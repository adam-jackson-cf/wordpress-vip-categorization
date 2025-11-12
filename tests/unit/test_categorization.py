"""Unit tests for categorization service."""

from unittest.mock import Mock, patch

from src.config import Settings
from src.models import WordPressContent
from src.services.categorization import CategorizationService


class TestCategorizationService:
    """Tests for categorization service."""

    def test_init(self, mock_settings: Settings, mock_supabase_client: Mock) -> None:
        """Test service initialization."""
        service = CategorizationService(mock_settings, mock_supabase_client)
        assert service.settings == mock_settings
        assert service.db == mock_supabase_client

    def test_create_categorization_prompt(
        self,
        mock_settings: Settings,
        mock_supabase_client: Mock,
        sample_wordpress_content: WordPressContent,
    ) -> None:
        """Test categorization prompt creation."""
        service = CategorizationService(mock_settings, mock_supabase_client)

        categories = ["Technology", "Business", "Health"]
        prompt = service.create_categorization_prompt(sample_wordpress_content, categories)

        assert "Technology, Business, Health" in prompt
        assert sample_wordpress_content.title in prompt
        assert "JSON" in prompt

    def test_prepare_batch_requests(
        self,
        mock_settings: Settings,
        mock_supabase_client: Mock,
        sample_wordpress_content: WordPressContent,
    ) -> None:
        """Test batch request preparation."""
        service = CategorizationService(mock_settings, mock_supabase_client)

        content_items = [sample_wordpress_content]
        categories = ["Technology", "Business"]

        requests = service.prepare_batch_requests(content_items, categories)

        assert len(requests) == 1
        assert requests[0]["custom_id"] == str(sample_wordpress_content.id)
        assert requests[0]["method"] == "POST"
        assert requests[0]["url"] == "/v1/chat/completions"
        assert "model" in requests[0]["body"]

    def test_create_batch_file(
        self,
        mock_settings: Settings,
        mock_supabase_client: Mock,
        sample_wordpress_content: WordPressContent,
    ) -> None:
        """Test batch file creation."""
        service = CategorizationService(mock_settings, mock_supabase_client)

        requests = [
            {
                "custom_id": str(sample_wordpress_content.id),
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {"model": "gpt-4o-mini"},
            }
        ]

        file_path = service.create_batch_file(requests)

        assert file_path.endswith(".jsonl")
        with open(file_path) as f:
            lines = f.readlines()
            assert len(lines) == 1

    @patch("src.services.categorization.openai.OpenAI")
    def test_submit_batch(
        self,
        mock_openai_class: Mock,
        mock_settings: Settings,
        mock_supabase_client: Mock,
        mock_openai_client: Mock,
        tmp_path,
    ) -> None:
        """Test batch submission."""
        mock_openai_class.return_value = mock_openai_client

        service = CategorizationService(mock_settings, mock_supabase_client)
        service.client = mock_openai_client

        # Create temp file
        test_file = tmp_path / "test.jsonl"
        test_file.write_text('{"test": "data"}\n')

        batch_id = service.submit_batch(str(test_file), "Test batch")

        assert batch_id == "test-batch-id"
        mock_openai_client.files.create.assert_called_once()
        mock_openai_client.batches.create.assert_called_once()

    @patch("src.services.categorization.openai.OpenAI")
    def test_get_batch_status(
        self,
        mock_openai_class: Mock,
        mock_settings: Settings,
        mock_supabase_client: Mock,
        mock_openai_client: Mock,
    ) -> None:
        """Test batch status retrieval."""
        mock_openai_class.return_value = mock_openai_client

        service = CategorizationService(mock_settings, mock_supabase_client)
        service.client = mock_openai_client

        status = service.get_batch_status("test-batch-id")

        assert status.batch_id == "test-batch-id"
        assert status.status == "completed"
        assert status.request_counts["total"] == 10

    def test_get_categories_from_taxonomy(
        self,
        mock_settings: Settings,
        mock_supabase_client: Mock,
        sample_taxonomy_page,
    ) -> None:
        """Test extracting categories from taxonomy."""
        mock_supabase_client.get_all_taxonomy.return_value = [
            sample_taxonomy_page,
        ]

        service = CategorizationService(mock_settings, mock_supabase_client)
        categories = service.get_categories_from_taxonomy()

        assert len(categories) == 1
        assert "Technology" in categories

"""Pytest fixtures and configuration."""

import os
from datetime import datetime
from unittest.mock import Mock
from uuid import uuid4

import pytest

from src.config import Settings
from src.data.supabase_client import SupabaseClient
from src.models import (
    CategorizationResult,
    MatchingResult,
    TaxonomyPage,
    WordPressContent,
)


@pytest.fixture
def mock_settings() -> Settings:
    """Create mock settings for testing."""
    os.environ.update(
        {
            "SUPABASE_URL": "https://test.supabase.co",
            "SUPABASE_KEY": "test-key",
            "OPENAI_API_KEY": "test-openai-key",
            "OPENAI_MODEL": "gpt-4o-mini",
            "WORDPRESS_VIP_SITES": "https://test1.com,https://test2.com",
            "TAXONOMY_FILE_PATH": "./data/taxonomy.csv",
        }
    )
    return Settings()


@pytest.fixture
def sample_wordpress_content() -> WordPressContent:
    """Create sample WordPress content."""
    return WordPressContent(
        id=uuid4(),
        url="https://example.com/sample-post",
        title="Sample Post Title",
        content="This is sample content about technology and innovation.",
        site_url="https://example.com",
        published_date=datetime(2024, 1, 1, 12, 0, 0),
        metadata={"type": "post", "wp_id": 123},
    )


@pytest.fixture
def sample_taxonomy_page() -> TaxonomyPage:
    """Create sample taxonomy page."""
    return TaxonomyPage(
        id=uuid4(),
        url="https://taxonomy.com/tech",
        category="Technology",
        description="Technology-related content",
        keywords=["technology", "innovation", "digital"],
    )


@pytest.fixture
def sample_categorization_result(
    sample_wordpress_content: WordPressContent,
) -> CategorizationResult:
    """Create sample categorization result."""
    return CategorizationResult(
        id=uuid4(),
        content_id=sample_wordpress_content.id,
        category="Technology",
        confidence=0.95,
        batch_id="test-batch-123",
    )


@pytest.fixture
def sample_matching_result(
    sample_taxonomy_page: TaxonomyPage,
    sample_wordpress_content: WordPressContent,
) -> MatchingResult:
    """Create sample matching result."""
    return MatchingResult(
        id=uuid4(),
        taxonomy_id=sample_taxonomy_page.id,
        content_id=sample_wordpress_content.id,
        similarity_score=0.85,
    )


@pytest.fixture
def mock_supabase_client(mocker) -> Mock:  # type: ignore[misc]
    """Create mock Supabase client."""
    mock_client = mocker.Mock(spec=SupabaseClient)

    # Mock common methods
    mock_client.get_all_content.return_value = []
    mock_client.get_all_taxonomy.return_value = []
    mock_client.get_all_matchings.return_value = []

    return mock_client


@pytest.fixture
def mock_openai_client(mocker) -> Mock:  # type: ignore[misc]
    """Create mock OpenAI client."""
    mock_client = mocker.Mock()

    # Mock embeddings
    mock_embedding_response = mocker.Mock()
    mock_embedding_response.data = [mocker.Mock(embedding=[0.1] * 1536)]
    mock_client.embeddings.create.return_value = mock_embedding_response

    # Mock chat completions
    mock_completion_response = mocker.Mock()
    mock_completion_response.choices = [
        mocker.Mock(
            message=mocker.Mock(
                content='{"category": "Technology", "confidence": 0.95, "reasoning": "test"}'
            )
        )
    ]
    mock_client.chat.completions.create.return_value = mock_completion_response

    # Mock batch operations
    mock_batch = mocker.Mock()
    mock_batch.id = "test-batch-id"
    mock_batch.status = "completed"
    mock_batch.created_at = datetime.now()
    mock_batch.completed_at = datetime.now()
    mock_batch.request_counts = mocker.Mock(total=10, completed=10, failed=0)
    mock_batch.metadata = {}
    mock_batch.output_file_id = "test-output-file"

    mock_client.batches.create.return_value = mock_batch
    mock_client.batches.retrieve.return_value = mock_batch

    # Mock file operations
    mock_file = mocker.Mock()
    mock_file.id = "test-file-id"
    mock_client.files.create.return_value = mock_file

    mock_file_content = mocker.Mock()
    mock_file_content.text = '{"custom_id": "test", "response": {"body": {"choices": [{"message": {"content": "{\\"category\\": \\"Technology\\", \\"confidence\\": 0.95}"}}]}}}'
    mock_client.files.content.return_value = mock_file_content

    return mock_client

"""Unit tests for WorkflowService."""

from unittest.mock import MagicMock, Mock, patch
from uuid import uuid4

import pytest

from src.config import Settings
from src.models import TaxonomyPage, WordPressContent
from src.services.workflow import WorkflowService


@pytest.fixture
def mock_settings() -> Settings:
    """Create mock settings."""
    settings = Mock(spec=Settings)
    settings.similarity_threshold = 0.85
    settings.enable_semantic_matching = True
    settings.enable_llm_categorization = True
    settings.semantic_candidate_limit = 25
    settings.llm_candidate_limit = 10
    settings.llm_candidate_min_score = 0.6
    settings.semantic_api_key = "test-semantic-key"
    settings.semantic_base_url = "https://semantic.example.com/v1"
    settings.semantic_embedding_model = "text-embedding-3-small"
    settings.llm_api_key = "test-llm-key"
    settings.llm_base_url = "https://llm.example.com/v1"
    settings.llm_model = "gpt-4o-mini"
    settings.llm_batch_timeout = 86400
    return settings


@pytest.fixture
def mock_db() -> MagicMock:
    """Create mock database client."""
    return MagicMock()


@pytest.fixture
def mock_matching_service() -> MagicMock:
    """Create mock matching service."""
    return MagicMock()


@pytest.fixture
def mock_categorization_service() -> MagicMock:
    """Create mock categorization service."""
    return MagicMock()


@pytest.fixture
def sample_taxonomy_pages() -> list[TaxonomyPage]:
    """Create sample taxonomy pages."""
    return [
        TaxonomyPage(
            id=uuid4(),
            url="https://example.com/page1",
            category="Technology",
            description="Tech content",
            keywords=["tech", "software"],
        ),
        TaxonomyPage(
            id=uuid4(),
            url="https://example.com/page2",
            category="Business",
            description="Business content",
            keywords=["business", "finance"],
        ),
    ]


@pytest.fixture
def sample_content_items() -> list[WordPressContent]:
    """Create sample WordPress content."""
    return [
        WordPressContent(
            id=uuid4(),
            url="https://site.com/post1",
            title="Tech Article",
            content="This is about technology",
            site_url="https://site.com",
        ),
        WordPressContent(
            id=uuid4(),
            url="https://site.com/post2",
            title="Business Article",
            content="This is about business",
            site_url="https://site.com",
        ),
    ]


@patch("src.services.categorization.DSPyOptimizer")
def test_workflow_service_initialization(
    mock_dspy_optimizer_class: MagicMock, mock_settings: Settings, mock_db: MagicMock
) -> None:
    """Test workflow service initialization."""
    service = WorkflowService(mock_settings, mock_db)

    assert service.settings == mock_settings
    assert service.db == mock_db
    assert service.matching_service is not None
    assert service.categorization_service is not None


def test_workflow_both_stages_enabled(
    mock_settings: Settings,
    mock_db: MagicMock,
    mock_matching_service: MagicMock,
    mock_categorization_service: MagicMock,
    sample_taxonomy_pages: list[TaxonomyPage],
    sample_content_items: list[WordPressContent],
) -> None:
    """Test workflow with both semantic and LLM stages enabled."""
    # Setup
    service = WorkflowService(
        mock_settings,
        mock_db,
        matching_service=mock_matching_service,
        categorization_service=mock_categorization_service,
    )

    # Mock matching service returns one unmatched taxonomy
    mock_matching_service.get_unmatched_taxonomy.return_value = [sample_taxonomy_pages[1]]
    mock_matching_service.match_taxonomy_to_content.return_value = [(sample_content_items[1], 0.95)]

    # Mock LLM categorization returns success
    mock_categorization_service.categorize_for_matching.return_value = {
        "matched": 1,
        "below_threshold": 0,
        "total": 1,
    }

    # Execute
    stats = service.run_matching_workflow(
        taxonomy_pages=sample_taxonomy_pages,
        content_items=sample_content_items,
        batch_mode=True,
    )

    # Verify
    assert mock_matching_service.match_all_taxonomy_batch.called
    mock_categorization_service.categorize_for_matching.assert_called_once()
    _, kwargs = mock_categorization_service.categorize_for_matching.call_args
    assert sample_taxonomy_pages[1].id in kwargs["candidate_map"]
    assert stats["semantic_matched"] == 1
    assert stats["llm_categorized"] == 1
    assert stats["needs_review"] == 0


def test_workflow_semantic_only(
    mock_settings: Settings,
    mock_db: MagicMock,
    mock_matching_service: MagicMock,
    mock_categorization_service: MagicMock,
    sample_taxonomy_pages: list[TaxonomyPage],
    sample_content_items: list[WordPressContent],
) -> None:
    """Test workflow with only semantic matching enabled."""
    # Disable LLM categorization
    mock_settings.enable_llm_categorization = False

    service = WorkflowService(
        mock_settings,
        mock_db,
        matching_service=mock_matching_service,
        categorization_service=mock_categorization_service,
    )

    # Mock matching service returns one unmatched taxonomy
    mock_matching_service.get_unmatched_taxonomy.return_value = [sample_taxonomy_pages[1]]

    # Execute
    stats = service.run_matching_workflow(
        taxonomy_pages=sample_taxonomy_pages,
        content_items=sample_content_items,
        batch_mode=True,
    )

    # Verify
    assert mock_matching_service.match_all_taxonomy_batch.called
    assert not mock_categorization_service.categorize_for_matching.called
    assert stats["semantic_matched"] == 1
    assert stats["llm_categorized"] == 0
    assert stats["needs_review"] == 1


def test_workflow_llm_only(
    mock_settings: Settings,
    mock_db: MagicMock,
    mock_matching_service: MagicMock,
    mock_categorization_service: MagicMock,
    sample_taxonomy_pages: list[TaxonomyPage],
    sample_content_items: list[WordPressContent],
) -> None:
    """Test workflow with only LLM categorization enabled."""
    # Disable semantic matching
    mock_settings.enable_semantic_matching = False

    service = WorkflowService(
        mock_settings,
        mock_db,
        matching_service=mock_matching_service,
        categorization_service=mock_categorization_service,
    )

    # Mock LLM categorization
    mock_categorization_service.categorize_for_matching.return_value = {
        "matched": 2,
        "below_threshold": 0,
        "total": 2,
    }
    mock_matching_service.match_taxonomy_to_content.return_value = [(sample_content_items[0], 0.8)]

    # Execute
    stats = service.run_matching_workflow(
        taxonomy_pages=sample_taxonomy_pages,
        content_items=sample_content_items,
        batch_mode=False,
    )

    # Verify
    assert not mock_matching_service.match_all_taxonomy.called
    assert not mock_matching_service.match_all_taxonomy_batch.called
    mock_categorization_service.categorize_for_matching.assert_called_once()
    _, kwargs = mock_categorization_service.categorize_for_matching.call_args
    assert kwargs["candidate_map"]
    assert stats["semantic_matched"] == 0
    assert stats["llm_categorized"] == 2
    assert stats["needs_review"] == 0


def test_workflow_both_disabled(
    mock_settings: Settings,
    mock_db: MagicMock,
    mock_matching_service: MagicMock,
    mock_categorization_service: MagicMock,
    sample_taxonomy_pages: list[TaxonomyPage],
    sample_content_items: list[WordPressContent],
) -> None:
    """Test workflow with both stages disabled."""
    # Disable both stages
    mock_settings.enable_semantic_matching = False
    mock_settings.enable_llm_categorization = False

    service = WorkflowService(
        mock_settings,
        mock_db,
        matching_service=mock_matching_service,
        categorization_service=mock_categorization_service,
    )

    # Execute
    stats = service.run_matching_workflow(
        taxonomy_pages=sample_taxonomy_pages,
        content_items=sample_content_items,
        batch_mode=True,
    )

    # Verify
    assert not mock_matching_service.match_all_taxonomy_batch.called
    assert not mock_categorization_service.categorize_for_matching.called
    assert stats["semantic_matched"] == 0
    assert stats["llm_categorized"] == 0
    assert stats["needs_review"] == 0
    assert stats["skipped"] == 2


def test_workflow_non_batch_mode(
    mock_settings: Settings,
    mock_db: MagicMock,
    mock_matching_service: MagicMock,
    mock_categorization_service: MagicMock,
    sample_taxonomy_pages: list[TaxonomyPage],
    sample_content_items: list[WordPressContent],
) -> None:
    """Test workflow with non-batch mode."""
    service = WorkflowService(
        mock_settings,
        mock_db,
        matching_service=mock_matching_service,
        categorization_service=mock_categorization_service,
    )

    # Mock matching service
    mock_matching_service.get_unmatched_taxonomy.return_value = []

    # Execute
    stats = service.run_matching_workflow(
        taxonomy_pages=sample_taxonomy_pages,
        content_items=sample_content_items,
        batch_mode=False,
    )

    # Verify non-batch mode was used
    assert mock_matching_service.match_all_taxonomy.called
    assert not mock_matching_service.match_all_taxonomy_batch.called
    assert stats["semantic_matched"] == len(sample_taxonomy_pages)
    assert stats["llm_categorized"] == 0
    assert stats["needs_review"] == 0


def test_run_managed_workflow_creates_run(
    mock_settings: Settings,
    mock_db: MagicMock,
    mock_matching_service: MagicMock,
    mock_categorization_service: MagicMock,
    sample_taxonomy_pages: list[TaxonomyPage],
    sample_content_items: list[WordPressContent],
) -> None:
    """Managed runs should create workflow metadata and persist checkpoints."""

    run_record = Mock(id=uuid4())
    mock_db.get_workflow_run_by_key.return_value = None
    mock_db.create_workflow_run.return_value = run_record
    mock_matching_service.get_unmatched_taxonomy.return_value = []
    mock_matching_service.match_all_taxonomy_batch.return_value = {
        page.id: Mock(content_id=uuid4()) for page in sample_taxonomy_pages
    }
    mock_matching_service.match_taxonomy_to_content.return_value = []
    mock_db.get_all_taxonomy.return_value = sample_taxonomy_pages
    mock_db.get_all_content.return_value = sample_content_items

    service = WorkflowService(
        mock_settings,
        mock_db,
        matching_service=mock_matching_service,
        categorization_service=mock_categorization_service,
    )

    stats = service.run_managed_workflow(run_key="run-123", batch_mode=True)

    assert stats["semantic_matched"] == len(sample_taxonomy_pages)
    mock_db.create_workflow_run.assert_called_once()
    mock_db.update_workflow_run.assert_called()
    mock_db.update_workflow_run.assert_called()

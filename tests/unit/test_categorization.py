"""Unit tests for categorization service."""

from unittest.mock import Mock, patch
from uuid import uuid4

from src.config import Settings
from src.models import TaxonomyPage, WordPressContent
from src.services.categorization import CategorizationService


class TestCategorizationService:
    """Tests for categorization service."""

    @patch("src.services.categorization.openai.OpenAI")
    def test_init(
        self, mock_openai_class: Mock, mock_settings: Settings, mock_supabase_client: Mock
    ) -> None:
        """Test service initialization patches OpenAI client."""

        mock_client = Mock()
        mock_openai_class.return_value = mock_client

        service = CategorizationService(mock_settings, mock_supabase_client)

        assert service.settings == mock_settings
        assert service.db == mock_supabase_client
        assert service.client == mock_client
        mock_openai_class.assert_called_once_with(
            api_key=mock_settings.llm_api_key,
            base_url=mock_settings.llm_base_url,
        )

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

    def test_parse_batch_results_success(
        self,
        mock_settings: Settings,
        mock_supabase_client: Mock,
    ) -> None:
        """parse_batch_results should return categorization models."""

        service = CategorizationService(mock_settings, mock_supabase_client)
        fake_id = str(uuid4())
        results = [
            {
                "custom_id": fake_id,
                "response": {
                    "body": {"choices": [{"message": {"content": '{"category": "Tech"}'}}]}
                },
            }
        ]

        parsed = service.parse_batch_results(results, "batch-123")

        assert parsed[0].category == "Tech"
        assert parsed[0].batch_id == "batch-123"

    def test_categorize_content_batch_async(
        self,
        mock_settings: Settings,
        mock_supabase_client: Mock,
        sample_wordpress_content: WordPressContent,
    ) -> None:
        """When wait=False no follow-up calls should occur."""

        service = CategorizationService(mock_settings, mock_supabase_client)
        service.prepare_batch_requests = Mock(return_value=[{}])
        service.create_batch_file = Mock(return_value="/tmp/file.jsonl")
        service.submit_batch = Mock(return_value="batch-42")
        service.wait_for_batch_completion = Mock()

        batch_id = service.categorize_content_batch(
            [sample_wordpress_content], ["Tech"], wait=False
        )

        assert batch_id == "batch-42"
        service.wait_for_batch_completion.assert_not_called()

    def test_categorize_for_matching_records_match(
        self,
        mock_settings: Settings,
        mock_supabase_client: Mock,
        sample_taxonomy_page: TaxonomyPage,
        sample_wordpress_content: WordPressContent,
    ) -> None:
        """Ensure matched taxonomy rows are persisted with stage metadata."""

        service = CategorizationService(mock_settings, mock_supabase_client)
        service._find_best_match_llm = Mock(
            return_value=(
                sample_wordpress_content,
                {
                    "topic_alignment": 0.9,
                    "intent_fit": 0.9,
                    "entity_overlap": 0.6,
                    "temporal_relevance": 0.5,
                    "decision": "accept",
                    "reasoning": "good match",
                },
            )
        )

        stats = service.categorize_for_matching(
            [sample_taxonomy_page],
            [sample_wordpress_content],
        )

        assert stats == {"matched": 1, "below_threshold": 0, "total": 1}
        mock_supabase_client.upsert_matching.assert_called_once()

    def test_accept_by_rubric_skips_entity_without_keywords(
        self,
        mocker,
        mock_settings: Settings,
        mock_supabase_client: Mock,
        sample_taxonomy_page: TaxonomyPage,
    ) -> None:
        """Entity threshold should only apply when taxonomy keywords exist."""

        mocker.patch("src.services.categorization.DSPyOptimizer")
        service = CategorizationService(mock_settings, mock_supabase_client)
        sample_taxonomy_page.keywords = []
        rubric = {
            "decision": "accept",
            "topic_alignment": 0.9,
            "intent_fit": 0.9,
            "entity_overlap": 0.1,
            "temporal_relevance": 0.9,
        }

        assert service._accept_by_rubric(sample_taxonomy_page, rubric)

    def test_accept_by_rubric_enforces_entity_with_keywords(
        self,
        mocker,
        mock_settings: Settings,
        mock_supabase_client: Mock,
        sample_taxonomy_page: TaxonomyPage,
    ) -> None:
        """When keywords are present, low entity overlap should fail."""

        mocker.patch("src.services.categorization.DSPyOptimizer")
        service = CategorizationService(mock_settings, mock_supabase_client)
        sample_taxonomy_page.keywords = ["ai"]
        rubric = {
            "decision": "accept",
            "topic_alignment": 0.9,
            "intent_fit": 0.9,
            "entity_overlap": 0.1,
            "temporal_relevance": 0.9,
        }

        assert not service._accept_by_rubric(sample_taxonomy_page, rubric)

    def test_accept_by_rubric_logs_warning_when_clamping(
        self,
        mocker,
        caplog,
        mock_settings: Settings,
        mock_supabase_client: Mock,
        sample_taxonomy_page: TaxonomyPage,
    ) -> None:
        """Clamping rubric scores outside [0, 1] should log a warning."""
        import logging

        mocker.patch("src.services.categorization.DSPyOptimizer")
        service = CategorizationService(mock_settings, mock_supabase_client)
        rubric = {
            "decision": "accept",
            "topic_alignment": 1.5,  # Above 1.0, will be clamped to 1.0
            "intent_fit": 0.75,  # Below threshold, will be rejected
            "entity_overlap": -0.1,  # Below 0.0, will be clamped to 0.0
            "temporal_relevance": 0.9,
        }

        with caplog.at_level(logging.WARNING):
            result = service._accept_by_rubric(sample_taxonomy_page, rubric)

        # Should reject because intent_fit (0.75) is below threshold (0.8)
        assert not result

        # Should have logged warnings for topic and entity
        assert "Clamped topic_alignment from 1.50 to 1.00" in caplog.text
        assert "Clamped entity_overlap from -0.10 to 0.00" in caplog.text
        assert "Clamped intent_fit" not in caplog.text

    def test_categorize_for_matching_marks_review(
        self,
        mock_settings: Settings,
        mock_supabase_client: Mock,
        sample_taxonomy_page: TaxonomyPage,
        sample_wordpress_content: WordPressContent,
    ) -> None:
        """Items failing rubric gate should become needs_human_review."""

        service = CategorizationService(mock_settings, mock_supabase_client)
        service._find_best_match_llm = Mock(return_value=(None, {}))

        stats = service.categorize_for_matching(
            [sample_taxonomy_page],
            [sample_wordpress_content],
        )

        assert stats == {"matched": 0, "below_threshold": 1, "total": 1}
        mock_supabase_client.upsert_matching.assert_called_once()

    @patch("src.services.categorization.Path.exists")
    @patch("src.services.categorization.DSPyOptimizer")
    def test_find_best_match_llm_parses_response(
        self,
        mock_dspy_optimizer_class: Mock,
        mock_path_exists: Mock,
        mock_settings: Settings,
        mock_supabase_client: Mock,
        sample_taxonomy_page: TaxonomyPage,
        sample_wordpress_content: WordPressContent,
    ) -> None:
        """Test that _find_best_match_llm correctly parses DSPy response into rubric."""
        mock_dspy_optimizer = Mock()
        # Selector returns index only (rubric ignored by new flow)
        mock_dspy_optimizer.predict_match.return_value = (0, {})
        # Judge returns the rubric used for gating
        judge_rubric = {
            "topic_alignment": 0.91,
            "intent_fit": 0.88,
            "entity_overlap": 0.7,
            "temporal_relevance": 0.4,
            "decision": "accept",
            "reasoning": "valid",
        }
        mock_dspy_optimizer.judge_candidate.return_value = judge_rubric
        mock_dspy_optimizer_class.return_value = mock_dspy_optimizer
        mock_path_exists.return_value = False

        service = CategorizationService(mock_settings, mock_supabase_client)

        best_match, rubric = service._find_best_match_llm(
            sample_taxonomy_page, [sample_wordpress_content]
        )

        assert best_match == sample_wordpress_content
        assert rubric == judge_rubric
        mock_dspy_optimizer.predict_match.assert_called_once_with(
            sample_taxonomy_page, [sample_wordpress_content]
        )
        mock_dspy_optimizer.judge_candidate.assert_called_once()

    @patch("src.services.categorization.Path.exists")
    @patch("src.services.categorization.DSPyOptimizer")
    def test_find_best_match_llm_handles_no_match(
        self,
        mock_dspy_optimizer_class: Mock,
        mock_path_exists: Mock,
        mock_settings: Settings,
        mock_supabase_client: Mock,
        sample_taxonomy_page: TaxonomyPage,
        sample_wordpress_content: WordPressContent,
    ) -> None:
        """Test that _find_best_match_llm handles no match case."""
        mock_dspy_optimizer = Mock()
        mock_dspy_optimizer.predict_match.return_value = (-1, {})
        mock_dspy_optimizer_class.return_value = mock_dspy_optimizer
        mock_path_exists.return_value = False

        service = CategorizationService(mock_settings, mock_supabase_client)

        best_match, rubric = service._find_best_match_llm(
            sample_taxonomy_page, [sample_wordpress_content]
        )

        assert best_match is None
        assert rubric == {}
        mock_dspy_optimizer.predict_match.assert_called_once_with(
            sample_taxonomy_page, [sample_wordpress_content]
        )

    def test_coerce_datetime_parses_timestamp(self, mock_settings, mock_supabase_client) -> None:
        service = CategorizationService(mock_settings, mock_supabase_client)
        expected = service._coerce_datetime(1_700_000_000)
        assert expected.year >= 2023

    def test_extract_request_count_handles_missing(
        self, mock_settings, mock_supabase_client
    ) -> None:
        service = CategorizationService(mock_settings, mock_supabase_client)
        count = service._extract_request_count({"completed": 5}, "completed")
        assert count == 5
        assert service._extract_request_count(None, "completed") == 0

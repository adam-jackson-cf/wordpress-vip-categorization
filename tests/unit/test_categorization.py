"""Unit tests for categorization service."""

import json
from pathlib import Path
from unittest.mock import Mock, patch
from uuid import uuid4

import pytest

from src.config import Settings
from src.models import MatchingResult, MatchStage, TaxonomyPage, WordPressContent
from src.optimization.dspy_optimizer import PromptContext
from src.services.categorization import (
    BatchRequestFile,
    CategorizationService,
    LLMBatchStats,
)


@pytest.fixture(autouse=True)
def _patch_dspy_optimizer(mocker):  # type: ignore[annotated-assignment]
    mock_optimizer = mocker.Mock()
    mock_optimizer.load_latest_model.return_value = None
    mock_optimizer.get_prompt_context.return_value = PromptContext(
        instructions=None,
        demonstrations=[],
    )
    return mocker.patch("src.services.categorization.DSPyOptimizer", return_value=mock_optimizer)


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

        categories = ["Veterinary Guidance", "Animal Health", "Pharmacovigilance"]
        prompt = service.create_categorization_prompt(sample_wordpress_content, categories)

        assert "Veterinary Guidance, Animal Health, Pharmacovigilance" in prompt
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
        categories = ["Veterinary Guidance", "Animal Health"]

        requests = service.prepare_batch_requests(content_items, categories)

        assert len(requests) == 1
        assert requests[0]["custom_id"] == str(sample_wordpress_content.id)
        assert requests[0]["method"] == "POST"
        assert requests[0]["url"] == "/v1/chat/completions"
        assert "model" in requests[0]["body"]

    @patch("src.services.categorization.openai.OpenAI")
    def test_format_content_section_includes_detection_cues(
        self,
        mock_openai_class: Mock,
        mock_settings: Settings,
        mock_supabase_client: Mock,
        sample_wordpress_content: WordPressContent,
    ) -> None:
        service = CategorizationService(mock_settings, mock_supabase_client)

        section = service._format_content_section(sample_wordpress_content)

        assert "Detected Audiences: veterinarians" in section
        assert "Detected Species: swine" in section

    @patch("src.services.categorization.openai.OpenAI")
    def test_prepare_llm_fallback_requests_carry_detection_hints(
        self,
        mock_openai_class: Mock,
        mock_settings: Settings,
        mock_supabase_client: Mock,
        sample_wordpress_content: WordPressContent,
        sample_taxonomy_page: TaxonomyPage,
    ) -> None:
        service = CategorizationService(mock_settings, mock_supabase_client)

        requests = service.prepare_llm_fallback_requests(
            [sample_wordpress_content],
            {sample_wordpress_content.id: [sample_taxonomy_page]},
        )

        assert len(requests) == 1
        prompt = requests[0]["body"]["messages"][1]["content"]
        assert "Detected Audiences: veterinarians" in prompt
        assert "Detected Species: swine" in prompt

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
        assert "Veterinary Guidance" in categories

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

    def test_prepare_llm_fallback_requests_includes_schema(
        self,
        mock_settings: Settings,
        mock_supabase_client: Mock,
        sample_wordpress_content: WordPressContent,
        sample_taxonomy_page: TaxonomyPage,
        sample_matching_result: MatchingResult,
    ) -> None:
        service = CategorizationService(mock_settings, mock_supabase_client)

        service.prompt_instructions = "DSPy optimized instruction"
        service.prompt_demonstrations = ["Example 1: taxonomy_content_type=Tech"]

        requests = service.prepare_llm_fallback_requests(
            [sample_wordpress_content],
            {sample_wordpress_content.id: [sample_taxonomy_page]},
            {sample_wordpress_content.id: sample_matching_result},
        )

        assert len(requests) == 1
        body = requests[0]["body"]
        assert body["response_format"]["type"] == "json_schema"
        assert "Candidate taxonomy pages" in body["messages"][1]["content"]
        assert "DSPy optimized instruction" in body["messages"][0]["content"]
        assert "Example 1" in body["messages"][1]["content"]

    def test_write_llm_request_files_chunks(
        self,
        mock_settings: Settings,
        mock_supabase_client: Mock,
        tmp_path: Path,
    ) -> None:
        mock_settings.llm_batch_chunk_size = 1
        mock_settings.llm_batch_artifact_dir = tmp_path
        service = CategorizationService(mock_settings, mock_supabase_client)

        requests = [
            {"custom_id": "a", "body": {}},
            {"custom_id": "b", "body": {}},
        ]

        files = service._write_llm_request_files(requests)

        assert len(files) == 2
        for artifact in files:
            assert artifact.path.exists()
        manifest = files[0].run_dir / "manifest.json"
        assert manifest.exists()

    def test_apply_llm_batch_results_accepts(
        self,
        mock_settings: Settings,
        mock_supabase_client: Mock,
        sample_wordpress_content: WordPressContent,
        sample_taxonomy_page: TaxonomyPage,
        sample_matching_result: MatchingResult,
    ) -> None:
        service = CategorizationService(mock_settings, mock_supabase_client)
        payload = {
            "decision": "accept",
            "taxonomy_id": str(sample_taxonomy_page.id),
            "taxonomy_url": str(sample_taxonomy_page.destination_url),
            "topic_alignment": 0.92,
            "intent_fit": 0.9,
            "entity_overlap": 0.7,
            "reasoning": "clear match",
        }
        results = [
            {
                "custom_id": str(sample_wordpress_content.id),
                "response": {
                    "body": {
                        "choices": [
                            {"message": {"content": json.dumps(payload)}}
                        ]
                    }
                },
            }
        ]

        stats = service.apply_llm_batch_results(
            results,
            "batch-1",
            taxonomy_lookup={sample_taxonomy_page.id: sample_taxonomy_page},
            semantic_results={sample_wordpress_content.id: sample_matching_result},
            content_lookup={sample_wordpress_content.id: sample_wordpress_content},
        )

        assert stats.to_dict() == {"matched": 1, "needs_review": 0, "total": 1}
        mock_supabase_client.bulk_upsert_matchings.assert_called_once()
        saved = mock_supabase_client.bulk_upsert_matchings.call_args[0][0][0]
        assert saved.match_stage == MatchStage.LLM_CATEGORIZED
        assert saved.taxonomy_id == sample_taxonomy_page.id

    def test_apply_llm_batch_results_handles_invalid_response(
        self,
        mock_settings: Settings,
        mock_supabase_client: Mock,
        sample_wordpress_content: WordPressContent,
        sample_matching_result: MatchingResult,
    ) -> None:
        service = CategorizationService(mock_settings, mock_supabase_client)
        results = [
            {
                "custom_id": str(sample_wordpress_content.id),
                "response": {"body": {"choices": []}},
            }
        ]

        stats = service.apply_llm_batch_results(
            results,
            "batch-bad",
            taxonomy_lookup={},
            semantic_results={sample_wordpress_content.id: sample_matching_result},
        )

        assert stats.to_dict() == {"matched": 0, "needs_review": 1, "total": 1}
        saved = mock_supabase_client.bulk_upsert_matchings.call_args[0][0][0]
        assert saved.match_stage == MatchStage.NEEDS_HUMAN_REVIEW

    def test_both_stages_failed_status(
        self,
        mock_settings: Settings,
        mock_supabase_client: Mock,
        sample_wordpress_content: WordPressContent,
        sample_taxonomy_page: TaxonomyPage,
    ) -> None:
        """Verify failed_at_stage='both_stages_failed' when semantic and LLM both fail."""
        service = CategorizationService(mock_settings, mock_supabase_client)

        # Create semantic result that failed at semantic stage
        semantic_result = MatchingResult(
            id=uuid4(),
            content_id=sample_wordpress_content.id,
            taxonomy_id=None,  # Semantic did not accept
            semantic_taxonomy_id=sample_taxonomy_page.id,
            semantic_similarity_score=0.65,  # Below threshold
            match_stage=MatchStage.NEEDS_LLM_REVIEW,
            failed_at_stage="semantic_matching",  # Failed at semantic stage
        )

        # LLM result that also rejects
        payload = {
            "decision": "review",  # LLM also rejects
            "taxonomy_id": str(sample_taxonomy_page.id),
            "taxonomy_url": str(sample_taxonomy_page.destination_url),
            "topic_alignment": 0.55,
            "intent_fit": 0.60,
            "entity_overlap": 0.40,
            "reasoning": "Not confident enough",
        }
        results = [
            {
                "custom_id": str(sample_wordpress_content.id),
                "response": {
                    "body": {
                        "choices": [
                            {"message": {"content": json.dumps(payload)}}
                        ]
                    }
                },
            }
        ]

        stats = service.apply_llm_batch_results(
            results,
            "batch-both-failed",
            taxonomy_lookup={sample_taxonomy_page.id: sample_taxonomy_page},
            semantic_results={sample_wordpress_content.id: semantic_result},
            content_lookup={sample_wordpress_content.id: sample_wordpress_content},
        )

        assert stats.to_dict() == {"matched": 0, "needs_review": 1, "total": 1}
        mock_supabase_client.bulk_upsert_matchings.assert_called_once()
        saved = mock_supabase_client.bulk_upsert_matchings.call_args[0][0][0]
        assert saved.match_stage == MatchStage.NEEDS_HUMAN_REVIEW
        assert saved.failed_at_stage == "both_stages_failed"  # Both stages failed

    def test_categorize_for_matching_waits_and_applies(
        self,
        mock_settings: Settings,
        mock_supabase_client: Mock,
        sample_wordpress_content: WordPressContent,
        sample_taxonomy_page: TaxonomyPage,
        tmp_path: Path,
    ) -> None:
        mock_settings.llm_batch_artifact_dir = tmp_path
        service = CategorizationService(mock_settings, mock_supabase_client)
        service.prepare_llm_fallback_requests = Mock(return_value=[{"custom_id": "1"}])
        artifact_path = tmp_path / "requests.jsonl"
        artifact_path.write_text("{}", encoding="utf-8")
        service._write_llm_request_files = Mock(
            return_value=[BatchRequestFile(path=artifact_path, run_dir=tmp_path, count=1)]
        )
        service.submit_batch = Mock(return_value="batch-123")
        service.wait_for_batch_completion = Mock()
        service.retrieve_batch_results = Mock(return_value=[{"custom_id": str(sample_wordpress_content.id)}])
        service.apply_llm_batch_results = Mock(
            return_value=LLMBatchStats(matched=1, needs_review=0, total=1)
        )

        stats = service.categorize_for_matching(
            [sample_wordpress_content],
            candidate_map={sample_wordpress_content.id: [sample_taxonomy_page]},
            fallback_taxonomy=[sample_taxonomy_page],
            semantic_results={
                sample_wordpress_content.id: MatchingResult(
                    taxonomy_id=None,
                    content_id=sample_wordpress_content.id,
                    semantic_taxonomy_id=sample_taxonomy_page.id,
                    semantic_similarity_score=0.6,
                    match_stage=MatchStage.NEEDS_LLM_REVIEW,
                )
            },
        )

        assert stats["matched"] == 1
        assert stats["batch_ids"] == ["batch-123"]
        service.apply_llm_batch_results.assert_called_once()

    def test_accept_by_rubric_skips_entity_without_keywords(
        self,
        mock_settings: Settings,
        mock_supabase_client: Mock,
        sample_taxonomy_page: TaxonomyPage,
    ) -> None:
        """Entity threshold should only apply when taxonomy keywords exist."""

        service = CategorizationService(mock_settings, mock_supabase_client)
        sample_taxonomy_page.key_topics = []
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
        mock_settings: Settings,
        mock_supabase_client: Mock,
        sample_taxonomy_page: TaxonomyPage,
    ) -> None:
        """When keywords are present, low entity overlap should fail."""

        service = CategorizationService(mock_settings, mock_supabase_client)
        sample_taxonomy_page.key_topics = ["ai"]
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
        caplog,
        mock_settings: Settings,
        mock_supabase_client: Mock,
        sample_taxonomy_page: TaxonomyPage,
    ) -> None:
        """Clamping rubric scores outside [0, 1] should log a warning."""
        import logging

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

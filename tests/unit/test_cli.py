"""CLI-level regression tests."""

from unittest.mock import Mock

from click.testing import CliRunner

from src.cli import cli
from src.models import MatchStage

runner = CliRunner()


def test_match_conflicting_filters_returns_error() -> None:
    """Passing multiple targeting flags should fail fast."""
    result = runner.invoke(
        cli,
        [
            "match",
            "--taxonomy-ids",
            "00000000-0000-0000-0000-000000000000",
            "--only-unmatched",
        ],
    )

    assert result.exit_code != 0
    assert "Use at most one" in result.output


def test_match_only_unmatched_triggers_llm_retry(
    mocker,
    sample_taxonomy_page,
) -> None:
    """`--only-unmatched --force-llm` should clear and rerun just the backlog."""

    mock_settings = mocker.Mock()
    mock_settings.similarity_threshold = 0.85
    mock_settings.llm_confidence_threshold = 0.9
    mock_settings.enable_semantic_matching = False
    mock_settings.enable_llm_categorization = True

    mocker.patch("src.cli.get_settings", return_value=mock_settings)

    mock_db = Mock()
    mocker.patch("src.cli.SupabaseClient", return_value=mock_db)

    mock_matching = mocker.Mock()
    mock_matching.get_unmatched_taxonomy.return_value = [sample_taxonomy_page]
    mocker.patch("src.cli.MatchingService", return_value=mock_matching)

    mock_workflow = mocker.Mock()
    mock_workflow.run_matching_workflow.return_value = {
        "semantic_matched": 0,
        "llm_categorized": 1,
        "needs_review": 0,
    }
    mocker.patch("src.cli.WorkflowService", return_value=mock_workflow)

    result = runner.invoke(
        cli,
        [
            "match",
            "--only-unmatched",
            "--skip-semantic",
            "--force-llm",
        ],
    )

    assert result.exit_code == 0
    mock_matching.get_unmatched_taxonomy.assert_called_once_with(0.85)
    mock_db.clear_matching_results.assert_called_once_with(
        [sample_taxonomy_page.id],
        [MatchStage.LLM_CATEGORIZED, MatchStage.NEEDS_HUMAN_REVIEW],
    )
    mock_workflow.run_matching_workflow.assert_called_once_with(
        taxonomy_pages=[sample_taxonomy_page],
        batch_mode=True,
    )

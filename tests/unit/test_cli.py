"""CLI-level regression tests."""

import tempfile
from pathlib import Path
from unittest.mock import Mock

from click.testing import CliRunner

from src.cli import cli
from src.models import MatchStage

runner = CliRunner()


def test_init_db_success_via_rpc(mocker) -> None:
    """init-db should call Supabase SQL RPC and succeed on 200."""

    mock_settings = mocker.Mock()
    mock_settings.supabase_url = "https://projectref.supabase.co"
    mock_settings.supabase_key = "test-key"
    mocker.patch("src.cli.get_settings", return_value=mock_settings)

    mock_response = mocker.Mock(status_code=200)
    mock_post = mocker.patch("src.cli._post_schema_sql_with_retry", return_value=mock_response)

    result = runner.invoke(cli, ["init-db"])

    assert result.exit_code == 0
    mock_post.assert_called_once()


def test_init_db_falls_back_on_rpc_failure(mocker) -> None:
    """init-db should fall back to manual instructions when RPC fails."""

    mock_settings = mocker.Mock()
    mock_settings.supabase_url = "https://projectref.supabase.co"
    mock_settings.supabase_key = "test-key"
    mocker.patch("src.cli.get_settings", return_value=mock_settings)

    mocker.patch("src.cli._post_schema_sql_with_retry", side_effect=Exception("network error"))

    result = runner.invoke(cli, ["init-db"])

    assert result.exit_code == 0
    assert "DATABASE INITIALIZATION REQUIRED" in result.output


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
    mock_workflow.run_matching_workflow.assert_called_once()
    kwargs = mock_workflow.run_matching_workflow.call_args.kwargs
    assert kwargs["taxonomy_pages"] == [sample_taxonomy_page]
    assert kwargs["batch_mode"] is True


def test_optimize_dataset_success(mocker, tmp_path) -> None:
    """Test optimize-dataset command with valid dataset."""
    # Create a test dataset file
    dataset_file = tmp_path / "dataset.csv"
    with open(dataset_file, "w", newline="", encoding="utf-8") as f:
        import csv

        writer = csv.DictWriter(
            f,
            fieldnames=[
                "taxonomy_category",
                "taxonomy_description",
                "taxonomy_keywords",
                "content_summaries",
                "best_match_index",
                "confidence",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "taxonomy_category": "Tech",
                "taxonomy_description": "Tech content",
                "taxonomy_keywords": "tech",
                "content_summaries": "0. Title: Post\n   URL: https://example.com\n   Preview: Content...",
                "best_match_index": "0",
                "confidence": "0.9",
            }
        )

    mock_settings = mocker.Mock()
    mock_settings.dspy_train_split_ratio = 0.2
    mock_settings.dspy_optimization_seed = 42
    mock_settings.dspy_num_threads = 1
    mock_settings.dspy_display_table = 5
    mock_settings.dspy_optimization_budget = "medium"
    mock_settings.dspy_reflection_model = ""
    mock_settings.llm_model = "gpt-4o-mini"
    mock_settings.llm_base_url = "https://api.openai.com/v1"
    mock_settings.llm_api_key = "test-key"

    mocker.patch("src.cli.get_settings", return_value=mock_settings)

    mock_db = Mock()
    mocker.patch("src.cli.SupabaseClient", return_value=mock_db)

    mock_optimizer = mocker.Mock()
    mock_examples = [mocker.Mock()] * 5
    mock_optimizer.load_training_dataset.return_value = mock_examples
    mock_optimized_model = mocker.Mock()
    mock_optimizer.optimize_with_dataset.return_value = mock_optimized_model
    mocker.patch("src.cli.DSPyOptimizer", return_value=mock_optimizer)

    result = runner.invoke(
        cli,
        [
            "optimize-dataset",
            "--dataset",
            str(dataset_file),
            "--optimizer",
            "gepa",
            "--budget",
            "light",
        ],
    )

    assert result.exit_code == 0
    mock_optimizer.load_training_dataset.assert_called_once_with(dataset_file)
    mock_optimizer.optimize_with_dataset.assert_called_once()


def test_optimize_dataset_file_not_found() -> None:
    """Test optimize-dataset command with non-existent dataset file."""
    result = runner.invoke(
        cli,
        [
            "optimize-dataset",
            "--dataset",
            "nonexistent.csv",
        ],
    )

    assert result.exit_code != 0
    # Click validates path existence before our code runs
    assert "does not exist" in result.output


def test_optimize_dataset_invalid_budget_combination() -> None:
    """Test optimize-dataset command with conflicting budget options."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        f.write("taxonomy_category,taxonomy_description,content_summaries,best_match_index\n")
        f.write("Tech,Tech content,Summaries,0\n")
        dataset_path = f.name

    try:
        result = runner.invoke(
            cli,
            [
                "optimize-dataset",
                "--dataset",
                dataset_path,
                "--optimizer",
                "gepa",
                "--budget",
                "light",
                "--max-full-evals",
                "10",
            ],
        )

        assert result.exit_code != 0
        assert "exactly one of" in result.output.lower()
    finally:
        Path(dataset_path).unlink()


def test_optimize_dataset_invalid_train_split() -> None:
    """Test optimize-dataset command with invalid train split."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        f.write("taxonomy_category,taxonomy_description,content_summaries,best_match_index\n")
        f.write("Tech,Tech content,Summaries,0\n")
        dataset_path = f.name

    try:
        result = runner.invoke(
            cli,
            [
                "optimize-dataset",
                "--dataset",
                dataset_path,
                "--train-split",
                "1.5",  # Invalid: > 1
            ],
        )

        assert result.exit_code != 0
        assert "Train split must be between 0 and 1" in result.output
    finally:
        Path(dataset_path).unlink()


def test_workflow_start_cli(mocker) -> None:
    mock_settings = mocker.Mock()
    mocker.patch("src.cli.get_settings", return_value=mock_settings)
    mock_db = mocker.Mock()
    mocker.patch("src.cli.SupabaseClient", return_value=mock_db)
    mock_service = mocker.Mock()
    mocker.patch("src.cli.WorkflowService", return_value=mock_service)

    result = runner.invoke(cli, ["workflow", "start", "--run-key", "demo", "--batch"])

    assert result.exit_code == 0
    mock_service.run_managed_workflow.assert_called_once_with(
        run_key="demo",
        batch_mode=True,
        resume=False,
    )


def test_workflow_resume_cli(mocker) -> None:
    mock_settings = mocker.Mock()
    mocker.patch("src.cli.get_settings", return_value=mock_settings)
    mock_db = mocker.Mock()
    mocker.patch("src.cli.SupabaseClient", return_value=mock_db)
    mock_service = mocker.Mock()
    mocker.patch("src.cli.WorkflowService", return_value=mock_service)

    result = runner.invoke(cli, ["workflow", "resume", "demo"])

    assert result.exit_code == 0
    mock_service.run_managed_workflow.assert_called_once_with(
        run_key="demo",
        batch_mode=True,
        resume=True,
    )


def test_workflow_status_cli_lists_runs(mocker) -> None:
    mock_settings = mocker.Mock()
    mocker.patch("src.cli.get_settings", return_value=mock_settings)
    mock_db = mocker.Mock()
    mock_db.list_workflow_runs.return_value = []
    mocker.patch("src.cli.SupabaseClient", return_value=mock_db)

    result = runner.invoke(cli, ["workflow", "status", "--limit", "5"])

    assert result.exit_code == 0
    mock_db.list_workflow_runs.assert_called_once_with(limit=5)

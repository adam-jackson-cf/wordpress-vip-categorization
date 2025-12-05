"""Unit tests for CLI facade helpers."""

from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import click
import pytest

from src.services.cli_facade import (
    FullRunOptions,
    MatchCommandOptions,
    OptimizeDatasetOptions,
    _describe_config,
    _parse_uuid_list,
    execute_full_run,
    execute_match,
    execute_optimize_dataset,
    load_taxonomy_urls,
    resolve_site_tokens_override,
)


def test_resolve_site_tokens_override_requires_tokens() -> None:
    default = [("https://example.com", "token")]
    with pytest.raises(click.BadParameter):
        resolve_site_tokens_override("https://missing.com", default)


def test_execute_optimize_dataset_budget_conflict_raises(monkeypatch) -> None:
    options = OptimizeDatasetOptions(
        dataset=Path("dummy.csv"),
        optimizer="gepa",
        budget="light",
        max_full_evals=10,
        max_metric_calls=None,
        train_split=None,
        output=Path("out.json"),
        save_program=False,
        num_threads=None,
        display_table=None,
        seed=None,
        report=None,
        no_save_config=False,
    )

    with pytest.raises(click.UsageError):
        execute_optimize_dataset(options)


def test_execute_optimize_dataset_invalid_split_raises() -> None:
    options = OptimizeDatasetOptions(
        dataset=Path("dummy.csv"),
        optimizer="gepa",
        budget=None,
        max_full_evals=None,
        max_metric_calls=None,
        train_split=1.5,
        output=Path("out.json"),
        save_program=False,
        num_threads=None,
        display_table=None,
        seed=None,
        report=None,
        no_save_config=False,
    )

    with pytest.raises(click.BadParameter):
        execute_optimize_dataset(options)


def test_load_taxonomy_urls_success(tmp_path: Path) -> None:
    csv_path = tmp_path / "taxonomy.csv"
    csv_path.write_text("Destination_URL\nhttps://foo\n \nhttps://bar \n", encoding="utf-8")

    urls = load_taxonomy_urls(csv_path)

    assert urls == ["https://foo", "https://bar"]


def test_load_taxonomy_urls_missing_column(tmp_path: Path) -> None:
    csv_path = tmp_path / "taxonomy.csv"
    csv_path.write_text("Other\nvalue\n", encoding="utf-8")

    with pytest.raises(click.ClickException):
        load_taxonomy_urls(csv_path)


def test_resolve_site_tokens_override_with_defaults() -> None:
    defaults = [("https://example.com", "token")]

    assert resolve_site_tokens_override(None, defaults) == defaults


def test_resolve_site_tokens_override_parses_entries() -> None:
    defaults = [("https://site.com", "existing-token")]

    result = resolve_site_tokens_override(
        "https://override.com|t1, https://site.com",
        defaults,
    )

    assert result == [
        ("https://override.com", "t1"),
        ("https://site.com", "existing-token"),
    ]


def test_parse_uuid_list_roundtrip() -> None:
    values = [uuid4(), uuid4()]

    parsed = _parse_uuid_list(",".join(str(value) for value in values))

    assert parsed == values


def test_parse_uuid_list_invalid_uuid() -> None:
    with pytest.raises(click.BadParameter):
        _parse_uuid_list("not-a-uuid")


def test_describe_config_outputs(capsys) -> None:
    settings = SimpleNamespace(
        enable_semantic_matching=True,
        enable_llm_categorization=False,
        similarity_threshold=0.91,
    )

    _describe_config(settings)

    captured = capsys.readouterr().out
    assert "Semantic matching: enabled" in captured
    assert "LLM categorization: disabled" in captured


def test_execute_match_conflicting_filters() -> None:
    options = MatchCommandOptions(
        taxonomy_ids=str(uuid4()),
        taxonomy_file=Path("taxonomy.csv"),
        only_unmatched=False,
    )

    with pytest.raises(click.UsageError):
        execute_match(options)


def test_execute_match_handles_missing_taxonomy(mocker, capsys) -> None:
    settings = SimpleNamespace(
        enable_semantic_matching=True,
        enable_llm_categorization=True,
        similarity_threshold=0.85,
    )
    mocker.patch("src.services.cli_facade.get_settings", return_value=settings)

    db = mocker.Mock()
    db.get_taxonomy_by_ids.return_value = []
    mocker.patch("src.services.cli_facade.SupabaseClient", return_value=db)
    mocker.patch("src.services.cli_facade.MatchingService")
    workflow_service = mocker.Mock()
    mocker.patch("src.services.cli_facade.WorkflowService", return_value=workflow_service)

    options = MatchCommandOptions(taxonomy_ids=str(uuid4()))

    execute_match(options)

    assert "No taxonomy rows matched" in capsys.readouterr().out
    workflow_service.run_matching_workflow.assert_not_called()


def test_execute_match_runs_full_flow(mocker, capsys) -> None:
    settings = SimpleNamespace(
        enable_semantic_matching=True,
        enable_llm_categorization=True,
        similarity_threshold=0.8,
    )
    mocker.patch("src.services.cli_facade.get_settings", return_value=settings)

    db = mocker.Mock()
    db.get_all_content.return_value = ["c1", "c2", "c3"]
    db.clear_matching_results.return_value = 2
    mocker.patch("src.services.cli_facade.SupabaseClient", return_value=db)

    matching_service = mocker.Mock()
    mocker.patch("src.services.cli_facade.MatchingService", return_value=matching_service)

    workflow_service = mocker.Mock()
    workflow_service.run_matching_workflow.return_value = {
        "semantic_matched": 2,
        "llm_categorized": 1,
        "needs_review": 3,
        "llm_batch_ids": ["batch-42"],
    }
    mocker.patch("src.services.cli_facade.WorkflowService", return_value=workflow_service)

    options = MatchCommandOptions(
        threshold=0.7,
        batch=False,
        skip_semantic=True,
        skip_llm=False,
        limit=2,
        taxonomy_ids=None,
        taxonomy_file=None,
        only_unmatched=False,
        force_semantic=True,
        force_llm=True,
    )

    execute_match(options)

    output = capsys.readouterr().out
    assert "Batch job IDs" in output
    assert db.get_all_content.called
    assert db.clear_matching_results.call_count == 2
    workflow_service.run_matching_workflow.assert_called_once()


def test_execute_full_run_happy_path(tmp_path: Path, mocker, capsys) -> None:
    taxonomy_file = tmp_path / "taxonomy.csv"
    taxonomy_file.write_text("Destination_URL\n", encoding="utf-8")
    output_path = tmp_path / "results.csv"

    base_settings = SimpleNamespace(
        taxonomy_file_path=taxonomy_file,
        enable_semantic_matching=True,
        enable_llm_categorization=True,
        similarity_threshold=0.85,
        get_wordpress_site_tokens=lambda: [("https://default.com", "token-default")],
    )
    match_settings = SimpleNamespace(
        enable_semantic_matching=True,
        enable_llm_categorization=False,
        similarity_threshold=0.91,
    )
    mock_get_settings = mocker.patch(
        "src.services.cli_facade.get_settings",
        side_effect=[base_settings, match_settings],
    )

    db = mocker.Mock()
    mocker.patch("src.services.cli_facade.SupabaseClient", return_value=db)

    ingestion_service = mocker.Mock()
    ingestion_service.load_taxonomy_from_csv.return_value = 7
    ingestion_service.ingest_wordpress_sites.return_value = 11
    mocker.patch("src.services.cli_facade.IngestionService", return_value=ingestion_service)

    exporter = mocker.Mock()
    exporter.export_to_csv.return_value = 9
    mocker.patch("src.services.cli_facade.CSVExporter", return_value=exporter)

    workflow_service = mocker.Mock()
    workflow_service.run_matching_workflow.return_value = {
        "semantic_matched": 5,
        "llm_categorized": 4,
        "needs_review": 1,
    }
    mocker.patch("src.services.cli_facade.WorkflowService", return_value=workflow_service)

    mocker.patch(
        "src.services.cli_facade.resolve_site_tokens_override",
        return_value=[("https://override.com", "token")],
    )

    options = FullRunOptions(
        taxonomy_file=taxonomy_file,
        sites="https://override.com|token",
        max_pages=5,
        since=datetime(2024, 1, 1),
        resume=True,
        threshold=0.91,
        batch=True,
        skip_semantic=False,
        skip_llm=True,
        output=output_path,
        min_similarity=0.6,
    )

    execute_full_run(options)

    out = capsys.readouterr().out
    assert "[4/4] Exporting combined results" in out
    ingestion_service.load_taxonomy_from_csv.assert_called_with(taxonomy_file)
    ingestion_service.ingest_wordpress_sites.assert_called_once()
    workflow_service.run_matching_workflow.assert_called_once_with(batch_mode=True)
    exporter.export_to_csv.assert_called_once_with(
        output_path, include_unmatched=True, min_similarity=0.6
    )
    assert mock_get_settings.call_args_list[1].kwargs["overrides"] == {
        "enable_llm_categorization": False,
        "similarity_threshold": 0.91,
    }
